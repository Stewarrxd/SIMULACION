import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta,UTC
import time
import os
import math
import json
import asyncio
import aiohttp
import websockets
import threading
import logging
import warnings
from collections import defaultdict
from queue import Queue, Empty, Full
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from collections import deque, defaultdict
from dataclasses import dataclass, field


# Import the corrected Binance API wrapper
from binance_api_mejorado import BinanceAPI

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Trading parameters (same as original strategy)
TAKE_PROFIT_PCT = 1
STOP_LOSS_PCT = 1.4
MAX_TRADE_DURATION_MINUTES = 100
FEE_RATE = 0.0005


@dataclass
class Trade:
    symbol: str
    trade_type: str
    entry_price: float
    entry_time: datetime
    tp: float
    sl: float
    quantity: float
    volatility_score: float
    confidence: float
    original_dir: str = ""  # <-- nueva: LONG/SHORT base (sin invertir)

    # --- NUEVO ---
    inverted: bool = False
    inv_ctx: Optional[List[float]] = field(default_factory=list)
    inv_score_delta: float = 0.0   # (ucb_invert - ucb_normal) al decidir


@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    price: float
    high: float
    low: float
    volume: float
    data_1m: pd.DataFrame
    data_5m: pd.DataFrame

class DataCache:
    """Cache inteligente para datos de mercado manteniendo la estrategia original"""
    def __init__(self, max_symbols: int = 40, max_candles: int = 100):
        self.max_symbols = max_symbols
        self.max_candles = max_candles
        self.cache_1m = {}  # symbol -> DataFrame
        self.cache_5m = {}  # symbol -> DataFrame
        self.last_update = {}  # symbol -> timestamp
        self.price_cache = {}  # symbol -> current_price
        self.lock = threading.Lock()

    def update_data(self, symbol: str, data_1m: pd.DataFrame, data_5m: pd.DataFrame):
        """Actualiza los datos en cache de forma thread-safe"""
        with self.lock:
            # Mantener solo las últimas N velas
            if len(data_1m) > self.max_candles:
                data_1m = data_1m.tail(self.max_candles).copy()
            if len(data_5m) > self.max_candles:
                data_5m = data_5m.tail(self.max_candles).copy()

            self.cache_1m[symbol] = data_1m
            self.cache_5m[symbol] = data_5m
            self.last_update[symbol] = datetime.now()

            # Actualizar precio actual
            if not data_1m.empty:
                self.price_cache[symbol] = data_1m['close'].iloc[-1]

    def get_data(self, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Obtiene datos del cache"""
        with self.lock:
            return (
                self.cache_1m.get(symbol, pd.DataFrame()).copy() if symbol in self.cache_1m else None,
                self.cache_5m.get(symbol, pd.DataFrame()).copy() if symbol in self.cache_5m else None
            )

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtiene el precio actual del cache"""
        with self.lock:
            return self.price_cache.get(symbol)

    def is_data_fresh(self, symbol: str, max_age_seconds: int = 60) -> bool:
        """Verifica si los datos están frescos"""
        with self.lock:
            if symbol not in self.last_update:
                return False
            age = (datetime.now() - self.last_update[symbol]).total_seconds()
            return age < max_age_seconds

class SymbolWebSocketPriceCache:
    """WebSocket para recibir precios en tiempo real por símbolo"""
    def __init__(self, symbols):
        self.symbols = [s.upper() for s in symbols]
        self.price_cache = {}  # symbol -> price
        self.tasks = {}
        self.lock = threading.Lock()
        self.running = False

    async def _ws_symbol(self, symbol):
        url = f"wss://fstream.binance.com/ws/{symbol.lower()}@markPrice"
        reconnect_delay = 1
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=None) as ws:
                    print(f"🟢 WS abierto para {symbol}")
                    while self.running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        price = float(data['p'])
                        with self.lock:
                            self.price_cache[symbol] = price
                        reconnect_delay = 1
            except Exception as e:
                print(f"🔴 Error WS {symbol}: {e}, reconectando en {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)

    def start(self):
        self.running = True
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()
        self._loop = loop
        for symbol in self.symbols:
            task = asyncio.run_coroutine_threadsafe(self._ws_symbol(symbol), loop)
            self.tasks[symbol] = task

    def stop(self):
        self.running = False
        time.sleep(2)
        for task in self.tasks.values():
            try:
                task.cancel()
            except Exception:
                pass

    def get_price(self, symbol):
        with self.lock:
            return self.price_cache.get(symbol.upper())


class OnlineLogisticRegression:
    def __init__(self, n_features: int, lr: float = 0.05, l2: float = 1e-4):
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -35, 35)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X):
        # X: (n_features,) o (n_samples, n_features)
        if X.ndim == 1:
            return float(self._sigmoid(np.dot(X, self.w) + self.b))
        z = X @ self.w + self.b
        return self._sigmoid(z)

    def fit(self, X, y, sample_weight=None, epochs: int = 5):
        # X: (n_samples, n_features), y: (n_samples,)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        if n == 0:
            return
        if sample_weight is None:
            sample_weight = np.ones(n, dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
        for _ in range(epochs):
            p = self.predict_proba(X)  # (n,)
            # gradientes con pesos y L2
            residual = (p - y) * sample_weight
            denom = np.sum(sample_weight) + 1e-9
            grad_w = (X.T @ residual) / denom + self.l2 * self.w
            grad_b = np.sum(residual) / denom
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b






class KlineWebSocketCache:
    """
    Mantiene klines por WebSocket para USDT-M Futures:
    - Suscribe a <symbol>@kline_<interval>
    - Actualiza en vivo un buffer (máx N velas) por (symbol, interval)
    - Permite leer DataFrames thread-safe con las velas (incluye velas en curso y cerradas)
    NOTA: Por WS (Futures) NO hay histórico: sólo acumularás desde que te conectas.
    """
    BASE_URL = "wss://fstream.binance.com/ws"

    def __init__(self,
                 pairs: Dict[str, List[str]],
                 max_candles: int = 1500,
                 include_open_candle: bool = True):
        """
        pairs: {'BTCUSDT': ['1m','5m'], 'ETHUSDT': ['1m']}
        max_candles: máximo de velas a mantener por (symbol, interval)
        include_open_candle: si True, la última vela (x=False) se mantiene/actualiza; si False, sólo cerradas.
        """
        self.pairs = {s.upper(): [i] if isinstance(i, str) else [x for x in i] for s, i in pairs.items()}
        self.max_candles = max_candles
        self.include_open = include_open_candle

        # buffers[(symbol, interval)] -> deque de dicts con campos de kline
        self.buffers: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=self.max_candles))
        self.lock = threading.Lock()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._tasks = {}
        self._thread: Optional[threading.Thread] = None

    async def _run_stream(self, symbol: str, interval: str):
        stream = f"{symbol.lower()}@kline_{interval}"
        url = f"{self.BASE_URL}/{stream}"
        backoff = 1
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=None) as ws:
                    # Opcional: enviar pings proactivos cada 20–30s
                    print(f"🟢 Kline WS conectado: {stream}")
                    backoff = 1
                    while self._running:
                        msg = await ws.recv()
                        data = json.loads(msg)

                        # Payload kline (Futures):
                        # data = {
                        #   "e":"kline","E":..., "s":"BTCUSDT",
                        #   "k":{"t":openTime, "T":closeTime, "s":"BTCUSDT", "i":"1m",
                        #        "f":firstTradeId, "L":lastTradeId, "o":"", "c":"", "h":"", "l":"",
                        #        "v":"", "n":trades, "x":isClosed, "q":"", "V":"", "Q":"", "B":"..."}
                        # }
                        if data.get("e") != "kline":
                            continue
                        k = data.get("k", {})
                        is_closed = bool(k.get("x", False))
                        if (not is_closed) and (not self.include_open):
                            # si no queremos vela abierta, saltar hasta que cierre
                            continue

                        row = {
                            "open_time": int(k["t"]),
                            "close_time": int(k["T"]),
                            "symbol": k["s"],
                            "interval": k["i"],
                            "open": float(k["o"]),
                            "high": float(k["h"]),
                            "low": float(k["l"]),
                            "close": float(k["c"]),
                            "volume": float(k["v"]),
                            "trades": int(k["n"]),
                            "quote_volume": float(k["q"]),
                            "taker_buy_volume": float(k["V"]),
                            "taker_buy_quote_volume": float(k["Q"]),
                            "is_closed": is_closed,
                        }

                        key = (symbol, interval)
                        with self.lock:
                            # si la vela (por open_time) ya existe, actualizar; si no, append
                            buf = self.buffers[key]
                            if len(buf) > 0 and buf[-1]["open_time"] == row["open_time"]:
                                buf[-1] = row
                            else:
                                buf.append(row)
            except Exception as e:
                print(f"🔴 Error WS {stream}: {e}. Reintentando en {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def start(self):
        if self._running:
            return
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        # lanzar una tarea por stream (símbolo-intervalo)
        for symbol, intervals in self.pairs.items():
            for interval in intervals:
                coro = self._run_stream(symbol, interval)
                task = asyncio.run_coroutine_threadsafe(coro, self._loop)
                self._tasks[(symbol, interval)] = task
        print("🚀 KlineWebSocketCache iniciado")

    def stop(self):
        self._running = False
        time.sleep(0.5)
        # Cancelar tasks
        for task in list(self._tasks.values()):
            try:
                task.cancel()
            except Exception:
                pass
        self._tasks.clear()
        # Cerrar loop
        try:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        print("🛑 KlineWebSocketCache detenido")

    def get_dataframe(self, symbol: str, interval: str, only_closed: bool = False) -> pd.DataFrame:
        """
        Devuelve un DataFrame con las velas acumuladas por WS.
        Si only_closed=True, filtra las velas con is_closed=True.
        """
        key = (symbol.upper(), interval)
        with self.lock:
            rows = list(self.buffers.get(key, deque()))
        if not rows:
            return pd.DataFrame(columns=[
                "timestamp","open","high","low","close","volume",
                "close_time","trades","quote_volume","taker_buy_volume","taker_buy_quote_volume","is_closed"
            ])
        df = pd.DataFrame(rows)
        if only_closed:
            df = df[df["is_closed"]].copy()
        # timestamps a datetime
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        return df[[
            "timestamp","open","high","low","close","volume",
            "close_time","trades","quote_volume","taker_buy_volume","taker_buy_quote_volume","is_closed"
        ]].reset_index(drop=True)

    def get_last_closed(self, symbol: str, interval: str) -> Optional[dict]:
        """Devuelve la última vela cerrada disponible por WS."""
        df = self.get_dataframe(symbol, interval, only_closed=True)
        if df.empty:
            return None
        return df.iloc[-1].to_dict()


class BanditLinUCB:
    """
    LinUCB por símbolo con 3 acciones: LONG, SHORT, HOLD.
    Mantiene diccionarios por acción: A (dxd), A_inv (dxd), b (d), n (count).
    Guarda/carga a JSON (listas) sin dependencias externas.
    """
    ACTIONS = ["LONG", "SHORT", "HOLD"]

    def __init__(self, d: int, alpha: float = 1.2):
        self.d = d
        self.alpha = float(alpha)
        # Por acción:
        self.A = {a: np.eye(d, dtype=np.float64) for a in self.ACTIONS}
        self.A_inv = {a: np.eye(d, dtype=np.float64) for a in self.ACTIONS}
        self.b = {a: np.zeros(d, dtype=np.float64) for a in self.ACTIONS}
        self.n = {a: 0 for a in self.ACTIONS}

    def _ucb_score(self, a: str, x: np.ndarray) -> float:
        A_inv = self.A_inv[a]
        theta = A_inv @ self.b[a]
        mean = float(theta @ x)
        # x^T A^-1 x (evita errores numéricos)
        s2 = float(x.T @ A_inv @ x)
        bonus = self.alpha * math.sqrt(max(s2, 0.0))
        return mean + bonus

    def choose(self, x: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Devuelve (accion, {scores por acción})"""
        scores = {a: self._ucb_score(a, x) for a in self.ACTIONS}
        action = max(scores.items(), key=lambda kv: kv[1])[0]
        return action, scores

    def update(self, action: str, x: np.ndarray, reward: float):
        """Actualiza A, b de la acción elegida con (x, reward)."""
        a = action
        A = self.A[a]
        b = self.b[a]
        # A <- A + x x^T ; b <- b + r x
        A += np.outer(x, x)
        b += reward * x
        # inversa estable (inv directa ok para d pequeño; para d grande: cho_factor/solve)
        self.A[a] = A
        self.b[a] = b
        try:
            self.A_inv[a] = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            # fallback: pseudo-inversa
            self.A_inv[a] = np.linalg.pinv(A)
        self.n[a] += 1

    # ---------- Persistencia ----------
    def to_dict(self) -> Dict:
        return {
            "d": self.d,
            "alpha": self.alpha,
            "A": {k: self.A[k].tolist() for k in self.ACTIONS},
            "A_inv": {k: self.A_inv[k].tolist() for k in self.ACTIONS},
            "b": {k: self.b[k].tolist() for k in self.ACTIONS},
            "n": dict(self.n),
        }

    @classmethod
    def from_dict(cls, data: Dict):
        obj = cls(d=int(data["d"]), alpha=float(data.get("alpha", 1.2)))
        for k in cls.ACTIONS:
            obj.A[k] = np.array(data["A"][k], dtype=np.float64)
            obj.A_inv[k] = np.array(data["A_inv"][k], dtype=np.float64)
            obj.b[k] = np.array(data["b"][k], dtype=np.float64)
            obj.n[k] = int(data["n"][k])
        return obj




# ============================================
# Order book WS cache (depth5@100ms)  # <<< NUEVO
# ============================================
class OrderBookWebSocketCache:
    BASE_URL = "wss://fstream.binance.com/ws"

    def __init__(self, symbols, levels=5, alpha_ema=0.2):
        self.symbols = [s.upper() for s in symbols]
        self.levels = int(levels)
        self.alpha = float(alpha_ema)
        self.metrics = {}     # symbol -> dict con métricas
        self.state = {}       # symbol -> {'imb_ema': ...}
        self.lock = threading.Lock()
        self._loop = None
        self._thread = None
        self._tasks = {}
        self._running = False

    async def _ws_symbol(self, symbol):
        stream = f"{symbol.lower()}@depth{self.levels}@100ms"
        url = f"{self.BASE_URL}/{stream}"
        backoff = 1
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=None) as ws:
                    print(f"🟢 OB WS conectado: {stream}")
                    backoff = 1
                    while self._running:
                        msg = await ws.recv()
                        data = json.loads(msg)

                        bids = [(float(p), float(q)) for p, q in data.get('b', []) if float(q) > 0]
                        asks = [(float(p), float(q)) for p, q in data.get('a', []) if float(q) > 0]
                        if not bids or not asks:
                            continue
                        bids.sort(key=lambda x: x[0], reverse=True)
                        asks.sort(key=lambda x: x[0])

                        bid1, qbid1 = bids[0]
                        ask1, qask1 = asks[0]
                        mid = 0.5 * (bid1 + ask1)
                        if mid <= 0:
                            continue
                        spread = ask1 - bid1
                        spread_bps = (spread / mid) * 1e4

                        sum_bid = sum(q for _, q in bids)
                        sum_ask = sum(q for _, q in asks)
                        denom = max(1e-12, (sum_bid + sum_ask))
                        imbalance = (sum_bid - sum_ask) / denom  # [-1,1]

                        micro = (ask1 * qbid1 + bid1 * qask1) / max(1e-12, (qbid1 + qask1))
                        micro_prem = (micro - mid) / mid  # fracción

                        st = self.state.get(symbol, {'imb_ema': 0.0})
                        prev = float(st.get('imb_ema', 0.0))
                        new_ema = self.alpha * imbalance + (1.0 - self.alpha) * prev
                        delta_imb = imbalance - prev
                        st['imb_ema'] = new_ema
                        self.state[symbol] = st

                        m = {
                            'best_bid': bid1,
                            'best_ask': ask1,
                            'mid': mid,
                            'spread_bps': spread_bps,
                            'imbalance': imbalance,
                            'microprice_premium': micro_prem,
                            'delta_imbalance': delta_imb,
                            'ts': datetime.now(UTC)
                        }
                        with self.lock:
                            self.metrics[symbol] = m
            except Exception as e:
                print(f"🔴 Error OB WS {symbol}: {e}. Reintentando en {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(60, backoff * 2)

    def start(self):
        if self._running:
            return
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        for s in self.symbols:
            coro = self._ws_symbol(s)
            task = asyncio.run_coroutine_threadsafe(coro, self._loop)
            self._tasks[s] = task
        print("🚀 OrderBookWebSocketCache iniciado")

    def stop(self):
        self._running = False
        time.sleep(0.5)
        for t in list(self._tasks.values()):
            try: 
                t.cancel()
            except: 
                pass
        self._tasks.clear()
        try:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
        except: 
            pass
        print("🛑 OrderBookWebSocketCache detenido")

    def get_metrics(self, symbol: str) -> dict:
        with self.lock:
            return dict(self.metrics.get(symbol.upper(), {}))




class LinUCBInversionPolicy:
    """
    Bandido contextual LinUCB para decidir si INVERTIR (1) o NO (0).
    Ligero, con olvido exponencial para adaptarse a temporadas.
    """
    def __init__(self, n_features: int, alpha: float = 1.0, l2: float = 5.0, decay: float = 0.995, min_obs: int = 20):
        self.n_features = n_features
        self.alpha = alpha
        self.l2 = l2
        self.decay = decay
        self.min_obs = min_obs
        self.A = {0: np.eye(n_features) * l2, 1: np.eye(n_features) * l2}
        self.b = {0: np.zeros(n_features, dtype=np.float64), 1: np.zeros(n_features, dtype=np.float64)}
        self.n = {0: 0, 1: 0}

    def ready(self) -> bool:
        return (self.n[0] + self.n[1]) >= self.min_obs

    def decide(self, x: np.ndarray) -> Tuple[int, float, float]:
        x = x.astype(np.float64, copy=False)
        scores = []
        for a in (0, 1):
            Ainv = np.linalg.pinv(self.A[a])
            theta = Ainv @ self.b[a]
            ucb = float(x @ theta + self.alpha * np.sqrt(max(1e-12, x @ Ainv @ x)))
            scores.append(ucb)
        action = int(np.argmax(scores))  # 0 = normal, 1 = invertir
        return action, scores[0], scores[1]

    def update(self, x: np.ndarray, action: int, reward: float):
        x = x.astype(np.float64, copy=False)
        # Olvido exponencial
        self.A[action] = self.decay * self.A[action] + np.outer(x, x)
        self.b[action] = self.decay * self.b[action] + reward * x
        self.n[action] += 1




class ParallelTradingBot:
    """
    Bot de trading paralelo que utiliza señales técnicas y volatilidad para abrir y cerrar operaciones.
    Puede operar en modo real (enviando órdenes a Binance) o en modo simulación (sin enviar órdenes).
    """
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False, simulate: bool = True):
        # Modo de simulación: si True, no se envían órdenes a Binance
        self.simulate = simulate
        self.api = BinanceAPI(api_key, api_secret, testnet=testnet)
        self.base_url = "https://fapi.binance.com"
        self.session = requests.Session()

        self.active_trades = {}
        self.completed_trades = []
        self.balance = 0
        self.consecutive_losses = 0
        self.running = False
        self.listen_key = self.get_listen_key()

        self.data_cache = DataCache()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.desired_tp = defaultdict(dict)
        self.desired_sl = defaultdict(dict)
        self.tp_sl_order_ids = defaultdict(lambda: {'tp': None, 'sl': None})

        self.price_queue = Queue(maxsize=1000)
        self.signal_queue = Queue(maxsize=50)
        self.exit_queue = Queue(maxsize=50)

        self.monitored_symbols = set()
        self.top_symbols = []
        self.trades_lock = threading.Lock()
        self.price_thread = None
        self.strategy_thread = None
        self._execution_thread = None
        self.monitor_thread = None

        # --- Parámetros de régimen de mercado (BTC) ---
        self.BTC_SYMBOL = "BTCUSDT"
        self.BTC_VOL_THRESHOLD = 0.01   # umbral de "alta volatilidad" para BTC (score ~2.5%)
        self.BTC_MOM_THR = 0.001        # 0.15% de momentum a 3 min para confirmar dirección




        # WebSocket de precios y caches
        self.ws_prices = None  # Precio en tiempo real por símbolo
        self.open_orders_cache = {}  # symbol -> list
        self.position_cache = {}     # symbol -> dict
        self.trade_open_history = defaultdict(list)  # symbol -> list of open timestamps
        self.signal_blocklist = {}  # symbol -> unblock_time (datetime)
        self.open_orders_cache_lock = threading.Lock() # Para acceso seguro al cache de órdenes abiertas

        # Lista para registrar señales que se convierten en operaciones. Se usará para simulaciones posteriores
        self.signals_log: List[Dict] = []

        # Diccionario para almacenar resultados de estrategias concurrentes
        self.strategy_results: Dict[str, Dict] = {}
        # Lista para hilos de estrategias concurrentes
        self.strategy_threads: List[threading.Thread] = []

        # Agrega en __init__ de ParallelTradingBot:
        self.ml_state = {}   # symbol -> {'model': OnlineLogisticRegression, 'mu':..., 'sigma':..., 'h': int, 'thr': float, 'n': int}

                # --- Señal de tendencia RSI (regresión) ---
        self.RSI_TREND_PRIMARY = True     # True: el RSI manda; False: solo fusiona con la señal ML
        self.RSI_SLOPE_WIN = 14           # velas 1m para la regresión
        self.RSI_SLOPE_THR = 0.25         # umbral de pendiente (puntos RSI/vela) para considerar tendencia
        self.RSI_TREND_MIN_R2 = 0.30      # calidad mínima del ajuste lineal

        self.RSI_HARD_OVERBOUGHT = 90     # extremo superior
        self.RSI_HARD_OVERSOLD  = 10      # extremo inferior
        # 'block' = no abrir en extremos y alta vol; 'follow' = solo seguir dirección obvia (90→SHORT, 10→LONG)
        self.RSI_EXTREME_MODE = 'follow'
        # al final del __init__ de ParallelTradingBot
        self.inv_policy = LinUCBInversionPolicy(n_features=12, alpha=1.2, l2=5.0, decay=0.995, min_obs=10)
        self.INVERSION_SCORE_MARGIN = 0.02     # margen mínimo (ucb_invert - ucb_normal)
        self.INVERSION_MAX_DRAWDOWN_STREAK = 3 # racha max de pérdidas con inversión antes de cooldown
        self.inversion_cooldown_until = datetime.min
        self._inv_losses_streak = 0

                # WebSocket de precios y caches
        self.ws_prices = None  # Precio en tiempo real por símbolo
        self.depth_ws = None   # <<< NUEVO: cache de libro (depth5)

                # === Bandido por símbolo ===
        self.bandit_states: Dict[str, BanditLinUCB] = {}     # symbol -> bandit
        self.bandit_meta: Dict[str, Dict] = {}               # symbol -> info (última acción, etc.)
        self.bandit_open: Dict[str, Dict] = {}               # symbol -> {'x':..., 'action':..., 'time':...}
        self.bandit_lock = threading.Lock()
        self.bandit_dir = os.path.join(os.getcwd(), "bandit_states")
        os.makedirs(self.bandit_dir, exist_ok=True)


    def _linreg_slope_r2(self, y: np.ndarray) -> Tuple[float, float]:
        """
        Devuelve (slope, r2) de una regresión lineal y ~ a + b*x sobre índices 0..n-1.
        slope en 'puntos RSI por vela'. r2 ∈ [0,1].
        """
        y = np.asarray(y, dtype=np.float64)
        n = y.size
        if n < 3:
            return 0.0, 0.0
        x = np.arange(n, dtype=np.float64)
        b1, b0 = np.polyfit(x, y, 1)  # y ≈ b1*x + b0
        y_hat = b1 * x + b0
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        return float(b1), float(r2)

# ================================
# 2) Métodos auxiliares ML dentro de ParallelTradingBot
# ================================
    def _safe_div(self, a, b):
        return a / b if b not in (0, 0.0, None) else 0.0

    def _std_scale(self,X, mu, sigma):
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        return (X - mu) / sigma

    def _rolling_atr_frac(self,df,window):
        # ATR% = ATR/close; vectorial
        h, l, c = df['high'], df['low'], df['close']
        prev_c = c.shift(1)
        tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
        atr = tr.rolling(window=window, min_periods=window).mean()
        return (atr / c).fillna(0.0)

    def _rolling_vol(self,df, window):
        ret = df['close'].pct_change()
        vol = ret.rolling(window=window, min_periods=window).std()
        return (vol * np.sqrt(1440)).fillna(0.0)  # anualiza aprox intradía 1m

# ================================
# 3) Construcción de features y dataset
# ================================
    def _ml_build_features(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """Crea features por vela (1m). Requiere columnas open/high/low/close/volume + EMA9/EMA26/RSI."""
        df = df_1m.copy()
        # Asegura indicadores mínimos
        if 'EMA9' not in df.columns or 'EMA26' not in df.columns or 'RSI' not in df.columns:
            df = self.calculate_indicators(df)

        # Retornos cortos
        df['ret_1'] = df['close'].pct_change(1)
        df['ret_3'] = df['close'].pct_change(3)
        df['ret_5'] = df['close'].pct_change(5)

        # Pendientes y gap de EMAs
        df['ema_fast'] = df['EMA9']
        df['ema_slow'] = df['EMA26']
        df['ema_fast_slope'] = df['ema_fast'].pct_change().fillna(0.0)
        df['ema_slow_slope'] = df['ema_slow'].pct_change().fillna(0.0)
        df['ema_gap'] = ((df['ema_fast'] - df['ema_slow']) / df['ema_slow']).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # RSI normalizado
        df['rsi_n'] = (df['RSI'] / 100.0).clip(0.0, 1.0)

        # Volatilidad y ATR%
        df['atr_frac'] = self._rolling_atr_frac(df, window=6)
        df['vol_std'] = self._rolling_vol(df, window=20)

        # Volatility score similar a tu métrica (aprox): mix de std y ATR%
        df['vol_score_ml'] = 0.6 * df['vol_std'] + 0.4 * df['atr_frac']

        # Selección final de features
        feats = [
            'ret_1','ret_3','ret_5',
            'rsi_n',
            'ema_fast_slope','ema_slow_slope','ema_gap',
            'atr_frac','vol_std','vol_score_ml'
        ]
        return df[feats + ['close', 'timestamp']].dropna().reset_index(drop=True)

    def _ml_prepare_training(self, df_feats: pd.DataFrame, horizon:int=5, thr:float=0.001):
        """
        Etiquetas: y=1 si ret_futuro >= +thr; y=0 si <= -thr; se ignoran casos intermedios.
        horizon=5 velas de 1m (~5 minutos).
        """
        if len(df_feats) <= horizon + 20:
            return None, None, None  # insuficiente

        closes = df_feats['close'].values
        X = df_feats.drop(columns=['close','timestamp']).values
        # futuro: retorno entre t -> t+h
        fut_ret = (closes[horizon:] - closes[:-horizon]) / closes[:-horizon]
        X_cut = X[:-horizon, :]
        y = np.full(len(X_cut), np.nan)
        y[fut_ret >= +thr] = 1.0
        y[fut_ret <= -thr] = 0.0
        mask = ~np.isnan(y)
        if mask.sum() < 30:
            return None, None, None
        X_train = X_cut[mask]
        y_train = y[mask].astype(np.float64)

        # Balanceo simple por pesos
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        w1 = (pos + neg) / (2.0 * max(pos, 1))
        w0 = (pos + neg) / (2.0 * max(neg, 1))
        sample_weight = np.where(y_train == 1, w1, w0).astype(np.float64)
        return X_train, y_train, sample_weight

    def _ml_get_or_train(self, symbol: str, df_1m: pd.DataFrame, horizon=5, thr=0.001):
        """Entrena/actualiza el modelo del símbolo y devuelve (state, ready_flag)."""
        df_feats = self._ml_build_features(df_1m)
        prep = self._ml_prepare_training(df_feats, horizon=horizon, thr=thr)
        if prep[0] is None:
            return None, False
        X_train, y_train, sw = prep

        # Escalado estándar por ventana
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        Xs = self._std_scale(X_train, mu, sigma)

        state = self.ml_state.get(symbol)
        if state is None:
            model = OnlineLogisticRegression(n_features=Xs.shape[1], lr=0.05, l2=1e-4)
            state = {'model': model, 'mu': mu, 'sigma': sigma, 'h': horizon, 'thr': thr, 'n': 0}
            self.ml_state[symbol] = state
        else:
            # refresca scaler a condiciones recientes
            state['mu'] = mu
            state['sigma'] = sigma

        # Entrenamiento online corto
        state['model'].fit(Xs, y_train, sample_weight=sw, epochs=6)
        state['n'] += len(y_train)
        return state, True


    # ---------- Bandit helpers ----------
    def _bandit_path(self, symbol: str) -> str:
        return os.path.join(self.bandit_dir, f"{symbol.upper()}.json")

    def _bandit_get_or_create(self, symbol: str, d: int, alpha: float = 1.2) -> BanditLinUCB:
        with self.bandit_lock:
            b = self.bandit_states.get(symbol)
            if b is not None and b.d == d:
                return b
            # si existe en disco, cargar
            path = self._bandit_path(symbol)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    b = BanditLinUCB.from_dict(data)
                    # si dims cambiaron por cambio de features, reinicializa
                    if b.d != d:
                        b = BanditLinUCB(d=d, alpha=alpha)
                except Exception:
                    b = BanditLinUCB(d=d, alpha=alpha)
            else:
                b = BanditLinUCB(d=d, alpha=alpha)
            self.bandit_states[symbol] = b
            return b

    def _bandit_save(self, symbol: str):
        with self.bandit_lock:
            b = self.bandit_states.get(symbol)
            if b is None:
                return
            path = self._bandit_path(symbol)
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(b.to_dict(), fh, ensure_ascii=False)
            except Exception as e:
                logger.debug(f"No se pudo guardar bandit {symbol}: {e}")

    def _bandit_store_open_ctx(self, symbol: str, x_vec: np.ndarray, action: str):
        # Guarda contexto/acción asociados a la posición abierta para actualizar al cierre
        self.bandit_open[symbol] = {
            "x": x_vec.astype(np.float64).tolist(),
            "action": action,
            "time": datetime.now().isoformat()
        }

    def _bandit_clear_open_ctx(self, symbol: str):
        if symbol in self.bandit_open:
            del self.bandit_open[symbol]

    def _reward_from_trade(self, trade: Trade, exit_price: float) -> float:
        """
        Recompensa escalar ~[-1,1]. Usa PnL neto/entrada con saturación tanh.
        """
        if trade.trade_type == "LONG":
            gross = (exit_price - trade.entry_price) * trade.quantity
        else:
            gross = (trade.entry_price - exit_price) * trade.quantity
        fees = (trade.entry_price + exit_price) * trade.quantity * FEE_RATE
        net = gross - fees
        # Normaliza por nocional de entrada
        base = max(trade.entry_price * trade.quantity, 1e-9)
        r = net / base
        # saturación para estabilidad
        return float(np.tanh(r))

    def _bandit_update_from_close(self, symbol: str, trade: Trade, exit_price: float):
        info = self.bandit_open.get(symbol)
        if not info:
            return
        x = np.array(info["x"], dtype=np.float64)
        action = str(info["action"])
        # d coherente con bandido
        b = self._bandit_get_or_create(symbol, d=len(x))
        reward = self._reward_from_trade(trade, exit_price)
        b.update(action, x, reward)
        self._bandit_save(symbol)
        self._bandit_clear_open_ctx(symbol)



    # --- Métodos auxiliares para integración WS ---
    def is_symbol_blocked(self, symbol):
        unblock_time = self.signal_blocklist.get(symbol)
        if not unblock_time:
            return False
        if datetime.now() >= unblock_time:
            # Ya venció el bloqueo, lo quitamos
            del self.signal_blocklist[symbol]
            return False
        return True

    def start_price_ws(self):
        if self.ws_prices:
            self.ws_prices.stop()
        symbols = list(self.monitored_symbols) if self.monitored_symbols else self.get_futures_symbols()[:40]
        if self.BTC_SYMBOL not in symbols:
            symbols.insert(0, self.BTC_SYMBOL)
        self.ws_prices = SymbolWebSocketPriceCache(symbols)
        self.ws_prices.start()
        print("🚀 WebSocket de precios iniciado")

    def start_depth_ws(self):  # <<< NUEVO
        symbols = list(self.monitored_symbols) if self.monitored_symbols else self.get_futures_symbols()[:40]
        if self.BTC_SYMBOL not in symbols:
            symbols.insert(0, self.BTC_SYMBOL)
        self.depth_ws = OrderBookWebSocketCache(symbols, levels=5, alpha_ema=0.2)
        self.depth_ws.start()
        print("📘 WebSocket de libro iniciado (depth5@100ms)")

    def get_ws_price(self, symbol):
        # Devuelve precio instantáneo (o None si aún no hay dato)
        return self.ws_prices.get_price(symbol) if self.ws_prices else None

    def get_open_orders_from_ws(self, symbol):
        return self.open_orders_cache.get(symbol, [])

    def get_position_from_ws(self, symbol):
        return self.position_cache.get(symbol, None)

    def get_listen_key(self):
        url = f"{self.base_url}/fapi/v1/listenKey"
        headers = {"X-MBX-APIKEY": self.api.client.API_KEY}
        resp = self.session.post(url, headers=headers)
        try:
            data = resp.json()
            if 'listenKey' not in data:
                raise ValueError(f"No se pudo obtener listenKey. Respuesta: {data}")
            return data['listenKey']
        except Exception as e:
            logger.error(f"❌ Error al obtener listenKey: {resp.text}")
            raise

    def get_futures_symbols(self) -> List[str]:
        """Obtiene símbolos de futuros activos"""
        try:
            url = f"{self.base_url}/fapi/v1/exchangeInfo"
            response = self.session.get(url, timeout=10)
            data = response.json()
            symbols = []
            for symbol_info in data['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['contractType'] == 'PERPETUAL' and
                    symbol_info['symbol'].endswith('USDT')):
                    symbols.append(symbol_info['symbol'])
            return symbols
        except Exception as e:
            logger.error(f"Error obteniendo símbolos: {e}")
            return []

    def get_24h_ticker_stats(self) -> pd.DataFrame:
        """Obtiene estadísticas 24h"""
        try:
            url = f"{self.base_url}/fapi/v1/ticker/24hr"
            response = self.session.get(url, timeout=10)
            data = response.json()
            df = pd.DataFrame(data)
            df['priceChangePercent'] = pd.to_numeric(df['priceChangePercent'])
            df['volume'] = pd.to_numeric(df['volume'])
            df['lastPrice'] = pd.to_numeric(df['lastPrice'])
            df = df[df['symbol'].str.endswith('USDT')]
            return df.sort_values('priceChangePercent', ascending=False)
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas 24h: {e}")
            return pd.DataFrame()

    def get_top_gainers_losers(self, df: pd.DataFrame, top_n: int = 18) -> Tuple[List[str], List[str]]:
        """Obtiene top gainers y losers"""
        min_volume = df['volume'].quantile(0.3)
        df_filtered = df[df['volume'] >= min_volume]
        top_gainers = df_filtered.head(top_n)['symbol'].tolist()
        top_losers = df_filtered.tail(top_n)['symbol'].tolist()
        return top_gainers, top_losers

    def update_monitored_symbols(self):
        """Actualiza símbolos monitoreados"""
        try:
            ticker_stats = self.get_24h_ticker_stats()
            if ticker_stats.empty:
                return
            top_gainers, top_losers = self.get_top_gainers_losers(ticker_stats, 18)
            new_symbols = set(top_gainers + top_losers)
                        # Asegurar que BTCUSDT siempre esté en la lista monitoreada
            new_symbols.add(self.BTC_SYMBOL)

            self.monitored_symbols = new_symbols
            self.top_symbols = list(new_symbols)
            logger.info(f"Actualizados símbolos monitoreados: {len(self.monitored_symbols)} símbolos")
        except Exception as e:
            logger.error(f"Error actualizando símbolos monitoreados: {e}")

    def get_klines(self, symbol: str, interval: str, limit: int = 720) -> pd.DataFrame:
        """Obtiene datos de velas"""
        try:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            response = self.session.get(url, params=params, timeout=6)
            data = response.json()
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                      'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                      'taker_buy_quote_volume', 'ignore']
            df = pd.DataFrame(data, columns=columns)
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.debug(f"Error obteniendo datos para {symbol}: {e}")
            return pd.DataFrame()

    def fetch_single_symbol_data(self, symbol: str) -> Optional[Tuple[str, pd.DataFrame, pd.DataFrame]]:
        """Obtiene datos de un símbolo usando la función de klines"""
        try:
            df_1m = self.get_klines(symbol, '1m', 720)
           
            df_5m = self.get_klines(symbol, '5m', 300)
            if df_1m.empty or df_5m.empty:
                return None
            return symbol, df_1m, df_5m
        except Exception as e:
            logger.debug(f"Error obteniendo datos para {symbol}: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores EMA y RSI"""
        if df.empty or len(df) < 26:
            return df
        df['EMA3'] = df['close'].ewm(span=5, adjust=False).mean()
        df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        delta = df['close'].diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = loss.abs()
        window = 6
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    def calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict:
        """Calcula métricas de volatilidad"""
        if df.empty or len(df) < 2:
            return {}
        df['returns'] = df['close'].pct_change()
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        volatility_std = df['returns'].std() * np.sqrt(1440)
        volatility_atr = df['true_range'].mean() / df['close'].mean()
        volatility_score = volatility_std * 0.6 + volatility_atr * 0.4
        return {
            'volatility_score': volatility_score,
            'current_price': df['close'].iloc[-1]
        }

    def _compute_btc_regime(self) -> Dict:
        """
        Calcula si BTC está en alta volatilidad y su dirección (UP/DOWN/FLAT) usando 1m y 5m.
        - Alta volatilidad: vol_score >= self.BTC_VOL_THRESHOLD
        - Dirección: cruce EMA9/EMA26 en 1m y 5m + momentum a 3m
        """
        try:
            df_btc_1m, df_btc_5m = self.data_cache.get_data(self.BTC_SYMBOL)
            if df_btc_1m is None or df_btc_1m.empty or len(df_btc_1m) < 60:
                res = self.fetch_single_symbol_data(self.BTC_SYMBOL)
                if res:
                    _, df_btc_1m, df_btc_5m = res
                    self.data_cache.update_data(self.BTC_SYMBOL, df_btc_1m, df_btc_5m)
                else:
                    return {'high_vol': False, 'dir': 'FLAT', 'vol_score': 0.0}

            df_btc_1m = self.calculate_indicators(df_btc_1m)
            if df_btc_5m is None or df_btc_5m.empty:
                df_btc_5m = self.get_klines(self.BTC_SYMBOL, '5m', 150)
            if not df_btc_5m.empty:
                df_btc_5m = self.calculate_indicators(df_btc_5m)

            vol_metrics_btc = self.calculate_volatility_metrics(df_btc_1m) or {}
            vol_score = float(vol_metrics_btc.get('volatility_score', 0.0) or 0.0)
            high_vol = vol_score >= self.BTC_VOL_THRESHOLD

            # Dirección por EMAs 1m y 5m + momentum 3 velas (3 min)
            if df_btc_1m.empty:
                return {'high_vol': high_vol, 'dir': 'FLAT', 'vol_score': vol_score}

            ema9_1m = df_btc_1m['EMA9'].iloc[-1] if 'EMA9' in df_btc_1m.columns else np.nan
            ema26_1m = df_btc_1m['EMA26'].iloc[-1] if 'EMA26' in df_btc_1m.columns else np.nan
            up_1m = (not np.isnan(ema9_1m)) and (not np.isnan(ema26_1m)) and (ema9_1m > ema26_1m)
            down_1m = (not np.isnan(ema9_1m)) and (not np.isnan(ema26_1m)) and (ema9_1m < ema26_1m)

            if df_btc_5m is not None and not df_btc_5m.empty:
                ema9_5m = df_btc_5m['EMA9'].iloc[-1] if 'EMA9' in df_btc_5m.columns else np.nan
                ema26_5m = df_btc_5m['EMA26'].iloc[-1] if 'EMA26' in df_btc_5m.columns else np.nan
                up_5m = (not np.isnan(ema9_5m)) and (not np.isnan(ema26_5m)) and (ema9_5m > ema26_5m)
                down_5m = (not np.isnan(ema9_5m)) and (not np.isnan(ema26_5m)) and (ema9_5m < ema26_5m)
            else:
                up_5m = down_5m = False

            ret3 = df_btc_1m['close'].pct_change(3).iloc[-1]
            mom_up = ret3 >= self.BTC_MOM_THR
            mom_dn = ret3 <= -self.BTC_MOM_THR

            if up_1m and up_5m and mom_up:
                direction = 'UP'
            elif down_1m and down_5m and mom_dn:
                direction = 'DOWN'
            else:
                direction = 'FLAT'

            return {'high_vol': high_vol, 'dir': direction, 'vol_score': vol_score}
        except Exception as e:
            logger.debug(f"Error computando régimen BTC: {e}")
            return {'high_vol': False, 'dir': 'FLAT', 'vol_score': 0.0}




    
    def analyze_symbol_signal(self, symbol: str) -> Dict:
        """Señal ML: logística online + filtros de volatilidad y alineación 5m."""
        try:
            df_1m, df_5m = self.data_cache.get_data(symbol)
            if df_1m is None or df_1m.empty or len(df_1m) < 60:
                result = self.fetch_single_symbol_data(symbol)
                if not result:
                    return {}
                _, df_1m, df_5m = result
                self.data_cache.update_data(symbol, df_1m, df_5m)

            # Indicadores base y 5m (para filtro de tendencia)
            df_1m = self.calculate_indicators(df_1m)
            if df_5m is None or df_5m.empty:
                df_5m = self.get_klines(symbol, '5m', 300)
            if not df_5m.empty:
                df_5m = self.calculate_indicators(df_5m)

            # Entrena/actualiza modelo del símbolo
            horizon = 5        # 5 minutos
            thr = 0.001        # 0.1% umbral para etiquetar
            state, ready = self._ml_get_or_train(symbol, df_1m, horizon=horizon, thr=thr)
            if not ready:
                return {}

            # Features actuales (última vela)
            feats_df = self._ml_build_features(df_1m)
            if feats_df.empty:
                return {}
            x_raw = feats_df.drop(columns=['close','timestamp']).iloc[-1].values.astype(np.float64)
            x = self._std_scale(x_raw, state['mu'], state['sigma'])

                        # === Bandit por símbolo (LinUCB) ===
            # x es el vector de features estandarizado (coherente con el modelo ML)
            bandit = self._bandit_get_or_create(symbol, d=len(x), alpha=1.2)
            bandit_action, bandit_scores = bandit.choose(x)


            # Probabilidad de subida a H=5 velas
            p_up = state['model'].predict_proba(x)

            # Filtro de volatilidad 1m
            # Reutiliza tu métrica original si quieres:
            vol_metrics = self.calculate_volatility_metrics(df_1m)
            vol_score = float(vol_metrics.get('volatility_score', 0.0) or 0.0)
            high_vol = vol_score > 0.02  # ~2% combinado; ajustable

            # Alineación 5m (no ir contra tendencia clara)
            ok_5m = True
            if not df_5m.empty:
                last5 = df_5m.iloc[-1]
                ema9_5m = last5.get('EMA9', np.nan)
                ema26_5m = last5.get('EMA26', np.nan)
                if not (np.isnan(ema9_5m) or np.isnan(ema26_5m)):
                    up_5m = ema9_5m > ema26_5m
                    down_5m = ema9_5m < ema26_5m
                else:
                    up_5m = down_5m = False
            else:
                up_5m = down_5m = False

            # Umbrales de decisión y dirección prevista
            # (más estrictos si la volatilidad es apenas suficiente)
            long_th = 0.60 if high_vol else 0.65
            short_th = 0.40 if high_vol else 0.35

            signal_type = None
            # tend. local por EMAs (1m) para no ir contra el microsesgo
            ema_fast_1m = df_1m['EMA9'].iloc[-1]
            ema_slow_1m = df_1m['EMA26'].iloc[-1]
            up_1m = ema_fast_1m > ema_slow_1m
            down_1m = ema_fast_1m < ema_slow_1m

                        # ---------------------------------------------
            # Señal por ML (dirección "logi_dir" opcional)
            # ---------------------------------------------
            logi_dir = None
            if p_up >= long_th and high_vol and (not down_5m) and up_1m:
                logi_dir = "LONG"
            elif p_up <= short_th and high_vol and (not up_5m) and down_1m:
                logi_dir = "SHORT"

            # ---------------------------------------------
            # Señal por RSI (regresión de tendencia)
            # ---------------------------------------------

                        # ---------------------------------------------
            # Señal por RSI (regresión de tendencia)
            # ---------------------------------------------
            rsi_dir = None
            rsi_slope = 0.0
            rsi_r2 = 0.0
            rsi_last = float(df_1m['RSI'].iloc[-1]) if 'RSI' in df_1m.columns else float('nan')

            if 'RSI' in df_1m.columns:
                rsi_win = self.RSI_SLOPE_WIN
                rsi_series = df_1m['RSI'].tail(rsi_win).dropna()

                if len(rsi_series) >= max(8, rsi_win // 2):  # menos estricto con longitud
                    rsi_slope, rsi_r2 = self._linreg_slope_r2(rsi_series.values)

                    # Señal fuerte: cumple ambos umbrales
                    if abs(rsi_slope) >= self.RSI_SLOPE_THR and rsi_r2 >= self.RSI_TREND_MIN_R2:
                        rsi_dir = "LONG" if rsi_slope > 0 else "SHORT"
                        rsi_strength = "strong"
                    # Fallback: pendiente clara pero r2 o slope no pasan umbral -> señal débil
                    elif abs(rsi_slope) >= 1e-3:
                        rsi_dir = "LONG" if rsi_slope > 0 else "SHORT"
                        rsi_strength = "weak"
                    else:
                        rsi_strength = "none"

                    # Log de diagnóstico (útil mientras ajustas)
                    logger.debug(
                        f"[RSI] len={len(rsi_series)} last={rsi_last:.2f} "
                        f"slope={rsi_slope:.4f} r2={rsi_r2:.2f} "
                        f"thr={self.RSI_SLOPE_THR} min_r2={self.RSI_TREND_MIN_R2} "
                        f"dir={rsi_dir} strength={rsi_strength}"
                    )
                else:
                    rsi_strength = "none"


            # -----------------------------------------------------
            # Regla de EXTREMOS con alta volatilidad (opcional)
            # -----------------------------------------------------
            extreme_blocked = False
            if high_vol and not np.isnan(rsi_last) and (rsi_last >= self.RSI_HARD_OVERBOUGHT or rsi_last <= self.RSI_HARD_OVERSOLD):
                if self.RSI_EXTREME_MODE == 'block':
                    # En extremos + alta vol, evitamos apertura
                    return {}
                else:
                    # 'follow': sólo se permite dirección "obvia" por nivel extremo
                    rsi_dir = "LONG" if rsi_last >= self.RSI_HARD_OVERBOUGHT else "SHORT"

            # -----------------------------------------------------
            # Fusión/selección final de dirección
            # -----------------------------------------------------
            # Modo: RSI manda si está activo y hay señal válida; si no, cae a ML; si ninguna, no hay operación
            if self.RSI_TREND_PRIMARY and rsi_dir is not None:
                signal_type = rsi_dir
            elif logi_dir is not None:
                signal_type = logi_dir
            elif rsi_dir is not None:
                signal_type = rsi_dir
            else:
                return {}

            # Reglas de coherencia con 5m: nunca ir contra cruce claro de 5m
            if signal_type == "LONG" and down_5m:
               return {}
            if signal_type == "SHORT" and up_5m:
               return {}

            # Precio actual
            current_price = self.get_ws_price(symbol) or float(df_1m['close'].iloc[-1])

            # Confianza combinada (RSI + ML)
            # Base ML ya calculada en 'confidence'. Creamos una confianza RSI y fusionamos.
            conf_rsi = 55.0
            conf_rsi += min(20.0, abs(rsi_slope) * 40.0)
            conf_rsi += min(15.0, rsi_r2 * 30.0)
            if (rsi_dir is not None) and ('rsi_strength' in locals()) and rsi_strength == "weak":
                conf_rsi -= 8.0   # penaliza señal débil
            conf_rsi = max(40.0, conf_rsi)

            # Bonus si RSI y ML coinciden
            if (rsi_dir is not None) and (logi_dir is not None) and (rsi_dir == logi_dir):
                conf_rsi += 5.0

             # Confianza: basada en separación de probas + fuerza de tendencia/volatilidad
            gap = abs(p_up - 0.5) * 2.0  # 0..1
            ema_gap = self._safe_div(abs(ema_fast_1m - ema_slow_1m), abs(ema_slow_1m))
            confidence = 50 + min(25, gap * 40 * (1.0 if high_vol else 0.7)) + min(20, ema_gap * 2000) + min(10, vol_score * 300)
            confidence = float(np.clip(confidence, 50, 98))

            if self.RSI_TREND_PRIMARY and rsi_dir is not None:
                confidence = float(np.clip(0.6 * conf_rsi + 0.4 * confidence, 50, 98))
            elif rsi_dir is not None and logi_dir is not None:
                confidence = float(np.clip(0.5 * conf_rsi + 0.5 * confidence, 50, 98))
            elif rsi_dir is not None and logi_dir is None:
                confidence = float(np.clip(conf_rsi, 50, 98))
            else:
                # sólo ML
                confidence = float(np.clip(confidence, 50, 98))

           


            

            rsi_1m = float(df_1m['RSI'].iloc[-1]) if 'RSI' in df_1m.columns else float('nan')
 # >>>>>>>>>>>>>> OVERRIDE POR RÉGIMEN BTC <<<<<<<<<<<<<<
            btc_regime = self._compute_btc_regime()
            btc_override = False
            if btc_regime['high_vol']:
                if btc_regime['dir'] == 'DOWN' and signal_type == 'LONG':
                    signal_type = 'SHORT'
                    btc_override = True
                elif btc_regime['dir'] == 'UP' and signal_type == 'SHORT':
                    signal_type = 'LONG'
                    btc_override = True
                # opcional: ajustar confianza si se alinea con BTC en alta volatilidad
                if btc_override:
                    confidence = float(min(98.0, confidence + 8.0))
                    logger.info(f"🔁 BTC override en {symbol}: dirBTC={btc_regime['dir']} vol={btc_regime['vol_score']:.3f}")

 
           
            base_dir = signal_type  # esta es la señal base sin invertir

            # -------- Filtro y boost usando libro de órdenes (depth5)  # <<< NUEVO (ARREGLADO)
            # Valores por defecto para incluir en el return aunque no haya libro aún
            ob_spread_bps = None
            ob_imbalance = None
            ob_microprice_bp = None
            ob_delta_imb = None

            try:
                ob = self.depth_ws.get_metrics(symbol) if self.depth_ws else {}
            except Exception:
                ob = {}

            # Considera "stale" si el tick del libro es viejo
            def _is_fresh(obm):
                ts = obm.get('ts')
                try:
                    return isinstance(ts, datetime) and (datetime.now(UTC) - ts).total_seconds() <= 3.0
                except Exception:
                    return False

            if ob and _is_fresh(ob):
                spread_bps = float(ob.get('spread_bps', 0.0) or 0.0)
                imb = float(ob.get('imbalance', 0.0) or 0.0)                # [-1,1]
                mp = float(ob.get('microprice_premium', 0.0) or 0.0)        # fracción
                d_imb = float(ob.get('delta_imbalance', 0.0) or 0.0)

                # Vetos básicos por microestructura
                if spread_bps > 8.0:
                    return {}  # spread demasiado grande: evitamos entrar

                need_imb = 0.10       # mínimo desequilibrio a favor
                need_mp  = 0.00002    # ~2 bps de microprice premium a favor

                if signal_type == 'LONG':
                    if not (imb > +need_imb and mp > +need_mp):
                        return {}
                else:  # SHORT
                    if not (imb < -need_imb and mp < -need_mp):
                        return {}

                # Boost de confianza si el flujo empuja fuerte
                boost = 0.0
                boost += min(8.0, max(0.0, (abs(imb) - 0.10) * 60.0))     # hasta +8
                boost += min(4.0, max(0.0, abs(mp) * 1e5 * 0.03))         # microprice (bps) * factor
                boost += min(3.0, max(0.0, abs(d_imb) * 100.0))           # aceleración

                confidence = float(np.clip(confidence + boost, 50, 98))

                # Guardar para el return y para LinUCB (vía _build_inversion_context)
                ob_spread_bps = spread_bps
                ob_imbalance = imb
                ob_microprice_bp = mp * 1e4    # pasar a bps
                ob_delta_imb = d_imb
            # -------- fin filtro libro  # <<< NUEVO (ARREGLADO)

            # Sugerencia base por ML:
            base_action = None
            if p_up >= long_th and high_vol and (not down_5m) and up_1m:
                base_action = "LONG"
            elif p_up <= short_th and high_vol and (not up_5m) and down_1m:
                base_action = "SHORT"

            # Combinar con bandit:
            # Regla simple y robusta:
            # - Si bandit dice HOLD -> respetar HOLD (no trade)
            # - Si ML no tiene acción, usar la del bandit si no es HOLD y hay volatilidad
            # - Si ambos tienen acción y difieren, usa la del bandit si su margen UCB es amplio.
            final_action = None
            if bandit_action == "HOLD":
                final_action = None
            elif base_action is None:
                final_action = bandit_action if high_vol else None
            else:
                if bandit_action == base_action:
                    final_action = base_action
                else:
                    # margen del bandit respecto a la segunda mejor
                    s_sorted = sorted(bandit_scores.items(), key=lambda kv: kv[1], reverse=True)
                    margin_ok = (len(s_sorted) >= 2) and ((s_sorted[0][1] - s_sorted[1][1]) > 0.02)
                    final_action = bandit_action if margin_ok else base_action

            if not final_action:
                return {}

            signal_type = final_action

            # Si BTC_OVERRIDE está activo, ya se cambió la dirección arriba.
            # Si NO está activo, puedes invertir la señal SOLO si es LONG.
            # Ejemplo: si la señal es LONG y btc_override es False, invierte a SHORT.
            # Si la señal es SHORT y btc_override es False, invierte a LONG.
            if not btc_override:
                if signal_type == 'LONG':
                    signal_type = 'SHORT' 
                elif signal_type == 'SHORT':
                    signal_type='LONG'



            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'base_dir': base_dir,
                'current_price': current_price,
                'confidence': confidence,
                'volatility_score': vol_score,
                # --- bandit trace ---
                'bandit_action': bandit_action,
                'bandit_scores': {k: float(v) for k, v in bandit_scores.items()},
                'bandit_context': x.tolist(),
                'p_up': float(p_up),
                'ema_fast': float(ema_fast_1m),
                'rsi': rsi_1m,
                'ema_slow': float(ema_slow_1m),
                'ob_spread_bps': ob_spread_bps,         # <<< NUEVO
                'ob_imbalance': ob_imbalance,           # <<< NUEVO
                'ob_microprice_bp': ob_microprice_bp,   # <<< NUEVO
                'ob_delta_imb': ob_delta_imb,           # <<< NUEVO
                    # <<< NUEVO
                # Dentro del dict de retorno de analyze_symbol_signal:
                'b_tc_dir': btc_regime.get('dir', 'FLAT'),
                'b_tc_vol': float(btc_regime.get('vol_score', 0.0)),
                'rsi_slope': float(rsi_slope),
                'rsi_r2': float(rsi_r2)
            }

            
        except Exception as e:
            logger.debug(f"Error analizando (ML) {symbol}: {e}")
            return {}




    def _build_inversion_context(self, signal: Dict) -> np.ndarray:  # <<< NUEVO (12 features)
        """
        Vector de 12 features normalizadas para el bandido:
        8 originales + 4 del libro:
        [p_up_centrado, vol_score, ema_ratio, rsi_norm, conf_norm, btc_dir, btc_vol, base_dir,
        microprice_bp, imbalance, spread_bps, delta_imb]
        """
        p_up = float(signal.get('p_up', 0.5))
        vol = float(signal.get('volatility_score', 0.0))
        ema_fast = float(signal.get('ema_fast', 0.0))
        ema_slow = float(signal.get('ema_slow', 1e-9))
        ema_ratio = ((ema_fast / ema_slow) - 1.0) if abs(ema_slow) > 1e-12 else 0.0
        rsi = float(signal.get('rsi', 50.0))
        conf = float(signal.get('confidence', 50.0)) / 100.0
        btc_dir = signal.get('b_tc_dir', 'FLAT')
        btc_dir_num = 1.0 if btc_dir == 'UP' else (-1.0 if btc_dir == 'DOWN' else 0.0)
        btc_vol = float(signal.get('b_tc_vol', 0.0))
        base_dir = 1.0 if signal.get('signal_type', 'LONG') == 'LONG' else -1.0

        # --- features de order book (con defaults robustos) ---
        ob_spread_bps = float(signal.get('ob_spread_bps', 0.0) or 0.0)
        ob_imb = float(signal.get('ob_imbalance', 0.0) or 0.0)               # [-1,1]
        ob_mp_bp = float(signal.get('ob_microprice_bp', 0.0) or 0.0)         # bps
        ob_dimb = float(signal.get('ob_delta_imb', 0.0) or 0.0)

        x = np.array([
            (p_up - 0.5) * 2.0,               # [-1,1]
            np.tanh(vol * 10.0),              # [0,1] aprox
            np.tanh(ema_ratio * 5.0),         # [-1,1]
            (rsi - 50.0) / 50.0,              # [-1,1]
            conf * 2.0 - 1.0,                 # [-1,1]
            btc_dir_num,                      # {-1,0,1}
            np.tanh(btc_vol * 10.0),          # [0,1]
            base_dir,                         # {-1,1}
            np.tanh(ob_mp_bp / 5.0),          # microprice premium en bps → [-1,1]
            ob_imb,                           # [-1,1]
            np.tanh(ob_spread_bps / 10.0),    # spread bps normalizado
            np.tanh(ob_dimb * 5.0)            # aceleración de imbalance
        ], dtype=np.float64)
        return x

    def _first_touch_winner(self, symbol: str, entry_time: datetime,
                            entry_price: float, tp_pct: float, sl_pct: float,
                            max_minutes: int = 180) -> Optional[str]:
        # Devuelve: 'BASE_WIN', 'INV_WIN' o None si no se pudo inferir.
        df_1m, _ = self.data_cache.get_data(symbol)
        if df_1m is None or df_1m.empty:
            return None

        seg = df_1m[df_1m['timestamp'] >= entry_time].copy()
        if seg.empty:
            return None
        seg = seg.head(max_minutes)

        # Niveles LONG base
        long_tp = entry_price * (1 + tp_pct/100.0)
        long_sl = entry_price * (1 - sl_pct/100.0)
        # Niveles SHORT invertido
        short_tp = entry_price * (1 - tp_pct/100.0)
        short_sl = entry_price * (1 + sl_pct/100.0)

        # Recorremos velas; el primer evento que aparezca decide
        for _, r in seg.iterrows():
            hi, lo = float(r['high']), float(r['low'])

            # Si base era LONG, su TP/SL son long_tp/long_sl; invertido (SHORT) tiene short_tp/short_sl
            base_tp_hit  = (hi >= long_tp)
            base_sl_hit  = (lo <= long_sl)
            inv_tp_hit   = (lo <= short_tp)
            inv_sl_hit   = (hi >= short_sl)

            # En el mundo real el orden intra-vela no se sabe; priorizamos “primera ocurrencia”
            # Considera empate raro como indeterminado
            if base_tp_hit and not (base_sl_hit or inv_tp_hit or inv_sl_hit):
                return 'BASE_WIN'
            if base_sl_hit and not (base_tp_hit or inv_tp_hit or inv_sl_hit):
                return 'INV_WIN'   # si la base toca SL primero, la invertida suele ser la mejor
            if inv_tp_hit and not (base_tp_hit or base_sl_hit or inv_sl_hit):
                return 'INV_WIN'
            if inv_sl_hit and not (base_tp_hit or base_sl_hit or inv_tp_hit):
                return 'BASE_WIN'

            # Casos múltiples en la misma vela -> seguimos a la siguiente (o decide por heurística si prefieres)

        return None

    def calculate_position_size(self, current_price: float) -> float:
        """Calcula el tamaño de la posición según la estrategia original"""
        base_notional = 5.2
        qty = base_notional / current_price
        min_notional = 5.0
        if qty * current_price < min_notional:
            qty = min_notional / current_price
        return qty

    def price_monitor_thread(self):
        """Hilo dedicado a monitorear precios y actualizar cache"""
        logger.info("🔄 Iniciando monitor de precios...")
        while self.running:
            try:
                if not self.monitored_symbols:
                    time.sleep(5)
                    continue
                symbol_chunks = list(self.monitored_symbols)
                batch_size = 10
                for i in range(0, len(symbol_chunks), batch_size):
                    if not self.running:
                        break
                    batch = symbol_chunks[i:i+batch_size]
                    futures = [self.executor.submit(self.fetch_single_symbol_data, symbol) for symbol in batch]
                    for future in as_completed(futures, timeout=10):
                        try:
                            result = future.result()
                            if result:
                                symbol, df_1m, df_5m = result
                                self.data_cache.update_data(symbol, df_1m, df_5m)
                        except Exception as e:
                            logger.debug(f"Error procesando future: {e}")
                    time.sleep(0.5)
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error en monitor de precios: {e}")
                time.sleep(5)

    def strategy_analysis_thread(self):
        """Hilo dedicado al análisis de estrategia"""
        logger.info("🧠 Iniciando análisis de estrategia...")
        while self.running:
            try:
                if not self.monitored_symbols:
                    time.sleep(5)
                    continue
                symbols_to_analyze = [
                s for s in list(self.monitored_symbols)  # snapshot
                if self.data_cache.is_data_fresh(s, 120)
            ]

                if not symbols_to_analyze:
                    time.sleep(2)
                    continue
                futures = [self.executor.submit(self.analyze_symbol_signal, symbol) 
                          for symbol in symbols_to_analyze]
                signals = []
                for future in as_completed(futures, timeout=15):
                    try:
                        signal = future.result()
                        if signal and signal.get('confidence', 0) > 40:
                            signals.append(signal)
                    except Exception as e:
                        logger.debug(f"Error en análisis de señal: {e}")
                signals.sort(key=lambda x: x['confidence'] * x['volatility_score'], reverse=True)
                max_signals = min(5, len(signals))
                for signal in signals[:max_signals]:
                    symbol = signal['symbol']
                    if self.is_symbol_blocked(symbol):
                        logger.info(f"🚫 {symbol}: Bloqueado temporalmente, no se agrega la señal a la cola.")
                        continue
                    try:
                        self.signal_queue.put_nowait(signal)
                    except Full:
                        logger.warning(f"⚠️ Cola de señales llena. Señal descartada: {signal['symbol']}")
                        break
                time.sleep(3)
            except Exception as e:
                logger.error(f"Error en análisis de estrategia: {e}")
                time.sleep(5)

    def open_trade(self, signal: Dict):
        """Abre nueva operación. En modo simulación no se envían órdenes a Binance"""
        try:
            symbol = signal['symbol']
            base_dir = signal.get('base_dir', signal['signal_type'])

            # ... dentro de open_trade, justo después de obtener 'symbol' y antes de calcular tp/sl:

# --- Decidir inversión con bandido contextual ---
            invert_now = False
            inv_score_delta = 0.0
            inv_ctx = self._build_inversion_context(signal)

            if datetime.now() >= self.inversion_cooldown_until:
                if self.inv_policy.ready():
                    action, u0, u1 = self.inv_policy.decide(inv_ctx)
                    inv_score_delta = float(u1 - u0)
                    if (action == 1) and (inv_score_delta > self.INVERSION_SCORE_MARGIN):
                        invert_now = True
                else:
                    # Warm-up: por prudencia no invertimos hasta min_obs
                    invert_now = False
            else:
                # Cooldown activo
                invert_now = False

            if invert_now:
                # Volteamos la dirección
                signal['signal_type'] = 'SHORT' if signal['signal_type'] == 'LONG' else 'LONG'
                logger.info(f"🔁 INVERSIÓN activada en {symbol} (Δscore={inv_score_delta:.4f})")
            else:
                if self.inv_policy.ready():
                    logger.debug(f"↔️ Sin inversión en {symbol} (Δscore={inv_score_delta:.4f})")




            now = datetime.now()
            """"""
            last_opens = self.trade_open_history[symbol]
            last_opens = [ts for ts in last_opens if (now - ts).total_seconds() < 120]
            if len(last_opens) >= 2:
                logger.info(f"🚫 {symbol}: Más de 2 aperturas en 2 minutos. Esperando a que baje la volatilidad.")
                self.signal_blocklist[symbol] = datetime.now() + timedelta(minutes=2)
                return
            last_opens.append(now)
            self.trade_open_history[symbol] = last_opens
            if symbol in self.active_trades:
                return
            

         # Mantener el precio que venía en la señal y solo sobrescribir si WS tiene dato
            ws_price = self.get_ws_price(symbol)
            if ws_price is not None:
                signal['current_price'] = float(ws_price)

            entry_price = signal.get('current_price')

            # Fallbacks si todavía no hay precio
            if entry_price is None:
                df_1m, _ = self.data_cache.get_data(symbol)
                if df_1m is not None and not df_1m.empty:
                    entry_price = float(df_1m['close'].iloc[-1])

            # Si sigue sin precio, salimos sin abrir
            if entry_price is None or entry_price <= 0:
                logger.warning(f"⚠️ {symbol}: sin precio válido para abrir operación. Se omite.")
                return


            trade_type = signal['signal_type']
            quantity = self.calculate_position_size(entry_price)
            if quantity <= 0:
                logger.warning(f"⚠️ Cantidad inválida para {symbol}, saltando operación")
                return
            if trade_type == "LONG":
                tp = entry_price * (1 + TAKE_PROFIT_PCT / 100)
                sl = entry_price * (1 - STOP_LOSS_PCT / 100)
            else:
                tp = entry_price * (1 - TAKE_PROFIT_PCT / 100)
                sl = entry_price * (1 + STOP_LOSS_PCT / 100)
            if not self.simulate:
                # En modo real se envían las órdenes
                if trade_type == "LONG":
                    result = self.api.open_long_position(symbol=symbol, quantity=quantity, leverage=50)
                else:
                    result = self.api.open_short_position(symbol=symbol, quantity=quantity, leverage=50)
                if not result:
                    logger.warning(f"❌ Falló la apertura de operación para {symbol}")
                    self.trade_open_history[symbol].pop()
                    return
                threading.Thread(
                    target=self.wait_and_set_tp_sl,
                    args=(symbol, trade_type, tp, sl),
                    daemon=True
                ).start()
            else:
                logger.info(f"🧪 (Simulado) Apertura de {trade_type} en {symbol} | Qty: {quantity:.4f} | TP: {tp:.4f} | SL: {sl:.4f}")
            side_key = trade_type.upper()
            self.desired_tp[symbol] = {side_key: tp, 'BOTH': tp}
            self.desired_sl[symbol] = {side_key: sl, 'BOTH': sl}

            trade_type = signal['signal_type']
            trade = Trade(
                        symbol=symbol,
                        trade_type=trade_type,
                        entry_price=entry_price,
                        original_dir=base_dir,           # <-- guarda la base
                        entry_time=datetime.now(),
                        tp=tp,
                        sl=sl,
                        quantity=quantity,
                        volatility_score=signal['volatility_score'],
                        confidence=signal['confidence'],
                        inverted=invert_now,                       # NUEVO
                        inv_ctx=inv_ctx.tolist(),                  # NUEVO
                        inv_score_delta=inv_score_delta            # NUEVO
                    )
            
                        # === Guarda contexto/acción del bandit para actualización al cierre ===
            if 'bandit_context' in signal and 'bandit_action' in signal:
                try:
                    bx = np.array(signal['bandit_context'], dtype=np.float64)
                    ba = str(signal['bandit_action'])
                    self._bandit_store_open_ctx(symbol, bx, ba)
                except Exception as _:
                    pass


            self.active_trades[symbol] = trade
            # Registrar la señal en el log de señales para análisis posterior, con datos relevantes
            self.signals_log.append({
                'symbol': symbol,
                'original_signal': base_dir,     # <-- esto era la base, no el trade ya invertido
                'timestamp': datetime.now(),
                'entry_price': entry_price,
                'tp': tp,
                'sl': sl,
                'quantity': quantity,
                'confidence': signal['confidence']
            })
            logger.info(f"🚀 NUEVA OPERACIÓN (simulada: {self.simulate}): {trade_type} {symbol}")
            logger.info(f"   Precio entrada: ${entry_price:.4f}")
            logger.info(f"   Take Profit: ${tp:.4f} (+{TAKE_PROFIT_PCT}%)")
            logger.info(f"   Stop Loss: ${sl:.4f} (-{STOP_LOSS_PCT}%)")
            logger.info(f"   Cantidad: {quantity:.6f}")
            logger.info(f"   Confianza: {signal['confidence']:.1f}%")
            logger.info(f"   RSI: {signal['rsi']:.2f}")
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"Error abriendo operación: {e}")

    def wait_and_set_tp_sl(self, symbol: str, trade_type: str, tp: float, sl: float, max_wait: int = 120):
        """Espera a que la posición se abra y coloca órdenes de TP/SL. En simulación se omite."""
        if self.simulate:
            logger.info(f"🧪 (Simulado) Configuración de TP/SL para {symbol} omitida.")
            return
        # Comportamiento original (abreviamos aquí; delega a la API real)
        try:
            waited = 0
            interval = 1
            logger.info(f"⏳ Esperando apertura real de posición en {symbol} para colocar TP/SL...")
            while waited < max_wait:
                positions = self.api.get_position_info(symbol)
                if not positions:
                    time.sleep(interval)
                    waited += interval
                    continue
                if isinstance(positions, dict):
                    positions_list = [positions]
                else:
                    positions_list = positions
                active_positions = [p for p in positions_list if abs(float(p.get('positionAmt', 0))) > 0]
                if not active_positions:
                    time.sleep(interval)
                    waited += interval
                    continue
                open_orders = self.api.get_open_orders(symbol) or []
                for pos in active_positions:
                    direction = trade_type.upper()
                    has_tp = any(
                        o['type'] in ['TAKE_PROFIT', 'TAKE_PROFIT_MARKET'] and o.get('positionSide') == 'BOTH'
                        for o in open_orders
                    )
                    has_sl = any(
                        o['type'] in ['STOP', 'STOP_MARKET'] and o.get('positionSide') == 'BOTH'
                        for o in open_orders
                    )
                    if not has_tp:
                        logger.info(f"⚠️ Colocando TP para {symbol} ({direction})...")
                        self.api.set_take_profit(symbol, tp, position_side=direction)
                    if not has_sl:
                        logger.info(f"⚠️ Colocando SL para {symbol} ({direction})...")
                        self.api.set_stop_loss(symbol, sl, position_side=direction)
                open_orders = self.api.get_open_orders(symbol) or []
                all_configured = True
                for pos in active_positions:
                    has_tp = any(
                        o['type'] in ['TAKE_PROFIT', 'TAKE_PROFIT_MARKET'] and o.get('positionSide') == 'BOTH'
                        for o in open_orders
                    )
                    has_sl = any(
                        o['type'] in ['STOP', 'STOP_MARKET'] and o.get('positionSide') == 'BOTH'
                        for o in open_orders
                    )
                    if not (has_tp and has_sl):
                        all_configured = False
                        break
                if all_configured:
                    logger.info(f"✅ TP/SL configurado correctamente para {symbol}")
                    return
                time.sleep(interval)
                waited += interval
            logger.warning(f"❌ Timeout configurando TP/SL para {symbol}")
        except Exception as e:
            logger.error(f"Error en wait_and_set_tp_sl: {e}")

    def check_exit_conditions_fast(self, symbol: str, trade: Trade) -> Optional[Dict]:
        """Salida inmediata usando el precio de WebSocket"""
        try:
            mark_price = self.get_ws_price(symbol)
            if mark_price is None:
                df_1m, _ = self.data_cache.get_data(symbol)
                if df_1m is None or df_1m.empty:
                    return None
                mark_price = df_1m.iloc[-1]['close']
            exit_reason = None
            exit_price = mark_price
            
            if trade.trade_type == "LONG":
                if mark_price >= trade.tp:
                    exit_reason = "Take Profit"
                elif mark_price <= trade.sl:
                    exit_reason = "Stop Loss"
            else:
                if mark_price <= trade.tp:
                    exit_reason = "Take Profit"
                elif mark_price >= trade.sl:
                    exit_reason = "Stop Loss"
            duration = datetime.now() - trade.entry_time
            if duration.total_seconds() / 60 >= MAX_TRADE_DURATION_MINUTES:
                exit_reason = "Timeout"
                exit_price = mark_price
            if exit_reason:
                return {
                    'symbol': symbol,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'timestamp': datetime.now()
                }
            return None
        except Exception as e:
            logger.debug(f"Error verificando salida para {symbol}: {e}")
            return None

    def trade_monitor_thread(self):
        """Monitor de operaciones para verificar condiciones de salida"""
        logger.info("👁️ Iniciando monitor de operaciones...")
        while self.running:
            try:
                with self.trades_lock:
                    active_symbols = list(self.active_trades.keys())
                if not active_symbols:
                    time.sleep(2)
                    continue
                futures = []
                for symbol in active_symbols:
                    with self.trades_lock:
                        if symbol in self.active_trades:
                            trade = self.active_trades[symbol]
                            futures.append(
                                self.executor.submit(self.check_exit_conditions_fast, symbol, trade)
                            )
                for future in as_completed(futures, timeout=10):
                    try:
                        exit_signal = future.result()
                        if exit_signal:
                            try:
                                self.exit_queue.put_nowait(exit_signal)
                            except Full:
                                logger.warning(f"⚠️ Cola de salidas llena. Señal de salida descartada: {exit_signal['symbol']}")
                    except Exception as e:
                        logger.debug(f"Error en verificación de salida: {e}")
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"Error en monitor de operaciones: {e}")
                time.sleep(2)

    def execution_thread(self):
        """Hilo de ejecución para procesar señales de entrada y salida"""
        logger.info("⚡ Iniciando hilo de ejecución...")
        while self.running:
            try:
                try:
                    signal = self.signal_queue.get(timeout=0.5)
                    self.process_entry_signal(signal)
                except Empty:
                    pass
                try:
                    exit_signal = self.exit_queue.get(timeout=0.1)
                    self.process_exit_signal(exit_signal)
                except Empty:
                    pass
            except Exception as e:
                logger.error(f"Error en hilo de ejecución: {e}")
                time.sleep(1)

    def process_entry_signal(self, signal: Dict):
        """Procesa señal de entrada"""
        try:
            max_op = 20
            symbol = signal['symbol']
            last_two = self.completed_trades[-1:]
            consec = sum(1 for t in last_two if t['symbol'] == symbol)
            if consec >= 10:
                logger.info(f"⚠️ No abrir más de 10 posiciones consecutivas en {symbol}. Saltando.")
                self.signal_blocklist[symbol] = datetime.now() + timedelta(minutes=2)
                return
            with self.trades_lock:
                if symbol in self.active_trades:
                    return
                if len(self.active_trades) >= max_op:
                    return
            self.open_trade(signal)
        except Exception as e:
            logger.error(f"Error procesando señal de entrada: {e}")

    def process_exit_signal(self, exit_signal: Dict):
        """Procesa señal de salida. En simulación no cierra posiciones reales."""
        try:
            symbol = exit_signal['symbol']
            exit_price = exit_signal['exit_price']
            exit_reason = exit_signal['exit_reason']
            with self.trades_lock:
                if symbol not in self.active_trades:
                    return
                trade = self.active_trades[symbol]
            if not self.simulate:
                self.api.close_all_positions(symbol)
                self.api.cancel_all_tp_sl_orders(symbol)
                self.api.cancel_all_limit_orders(symbol)
                self.refresh_open_orders_cache(symbol)
                time.sleep(1)
                self.refresh_open_orders_cache(symbol)
            else:
                logger.info(f"🧪 (Simulado) Cierre de {symbol} por {exit_reason} a ${exit_price:.4f}")
            if trade.trade_type == "LONG":
                roi = ((exit_price - trade.entry_price) / trade.entry_price) * 100
                pnl = (exit_price - trade.entry_price) * trade.quantity
            else:
                roi = ((trade.entry_price - exit_price) / trade.entry_price) * 100
                pnl = (trade.entry_price - exit_price) * trade.quantity
            entry_fee = trade.entry_price * trade.quantity * FEE_RATE
            exit_fee = exit_price * trade.quantity * FEE_RATE
            total_fees = entry_fee + exit_fee
            result = pnl - total_fees
            self.balance += result
            if result > 0:
                self.consecutive_losses = max(self.consecutive_losses - 1, 0)
            else:
                self.consecutive_losses += 1

                        # === Actualización de bandit por símbolo al cierre ===
            try:
                self._bandit_update_from_close(symbol, trade, exit_price)
            except Exception as _:
                logger.debug(f"No se pudo actualizar bandit para {symbol}")


            # --- Actualización de bandido con la recompensa de esta operación ---
            # justo después de calcular 'result' y antes de construir completed_trade
            try:
                if trade.inv_ctx:
                    x = np.array(trade.inv_ctx, dtype=np.float64)

                    # 1) Update “on-policy” (la acción que realmente tomaste), como ya tienes:
                    action_taken = 1 if trade.inverted else 0
                    notional = max(1e-9, trade.entry_price * trade.quantity)
                    scale = max(1e-6, TAKE_PROFIT_PCT / 100.0)
                    reward_taken = float(np.tanh((result / notional) / scale))
                    self.inv_policy.update(x, action_taken, reward_taken)

                    # 2) Update contrafactual (aprendizaje real de cuándo revertir):
                    winner = self._first_touch_winner(
                        symbol=trade.symbol,
                        entry_time=trade.entry_time,
                        entry_price=trade.entry_price,
                        tp_pct=TAKE_PROFIT_PCT,
                        sl_pct=STOP_LOSS_PCT
                    )
                    if winner is not None:
                        # acción 0 = NO invertir (seguir base), acción 1 = invertir
                        reward_base = +1.0 if winner == 'BASE_WIN' else -1.0
                        reward_inv  = +1.0 if winner == 'INV_WIN' else -1.0
                        self.inv_policy.update(x, 0, reward_base)
                        self.inv_policy.update(x, 1, reward_inv)
            except Exception as _e:
                logger.debug(f"Counterfactual update failed: {_e}")


            completed_trade = {
                'symbol': symbol,
                'type': trade.trade_type,
                'entry_time': trade.entry_time,
                'exit_time': datetime.now(),
                'entry_price': trade.entry_price,
                'exit_price': exit_price,
                'roi': roi,
                'result': result,
                'reason': exit_reason,
                'quantity': trade.quantity,
                'fees': total_fees,
                'confidence': trade.confidence
            }
            self.completed_trades.append(completed_trade)
            status_emoji = "✅" if result > 0 else "❌"
            logger.info(f"{status_emoji} OPERACIÓN CERRADA (simulada: {self.simulate}): {trade.trade_type} {symbol}")
            logger.info(f"   Precio salida: ${exit_price:.4f}")
            logger.info(f"   ROI: {roi:.2f}%")
            logger.info(f"   Resultado: ${result:.2f}")
            logger.info(f"   Motivo: {exit_reason}")
            logger.info(f"   Balance total: ${self.balance:.2f}")
            logger.info("-" * 50)
            with self.trades_lock:
                if symbol in self.active_trades:
                    del self.active_trades[symbol]
        except Exception as e:
            logger.error(f"Error procesando señal de salida: {e}")

    async def _position_ws(self, listen_key):
        url = f"wss://fstream.binance.com/ws/{listen_key}"
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get('e') == 'ACCOUNT_UPDATE':
                            await self._on_account_update(data['a'])
                        elif data.get('e') == 'ORDER_TRADE_UPDATE':
                            await self._on_order_trade_update(data['o'])

    async def _on_account_update(self, account_data: dict):
        """Manejador de ACCOUNT_UPDATE. En simulación no coloca TP/SL reales"""
        if self.simulate:
            for pos in account_data.get('P', []):
                symbol = pos.get('s')
                if symbol:
                    self.refresh_open_orders_cache(symbol)
            return
        try:
            for pos in account_data.get('P', []):
                amt = float(pos.get('pa', 0))
                if abs(amt) <= 0:
                    continue
                symbol = pos.get('s')
                position_side_field = pos.get('ps', 'BOTH')
                tp = None
                sl = None
                if symbol in self.desired_tp:
                    tp = self.desired_tp[symbol].get(position_side_field) or self.desired_tp[symbol].get('BOTH')
                if symbol in self.desired_sl:
                    sl = self.desired_sl[symbol].get(position_side_field) or self.desired_sl[symbol].get('BOTH')
                if tp is None or sl is None:
                    logger.warning(f"⚠️ No se encontraron TP/SL para {symbol} {position_side_field}")
                    continue
                direction = 'LONG' if tp > sl else 'SHORT'
                open_orders = self.open_orders_cache.get(symbol, [])
                has_tp = any(
                    o['type'] in ['TAKE_PROFIT', 'TAKE_PROFIT_MARKET'] and o.get('positionSide') == 'BOTH'
                    for o in open_orders
                )
                has_sl = any(
                    o['type'] in ['STOP', 'STOP_MARKET'] and o.get('positionSide') == 'BOTH'
                    for o in open_orders
                )
                if not has_tp:
                    logger.info(f"📤 Colocando TP para {symbol} @ {tp} (dir: {direction})")
                    self.api.set_take_profit(symbol, tp, position_side=direction)
                if not has_sl:
                    logger.info(f"📤 Colocando SL para {symbol} @ {sl} (dir: {direction})")
                    self.api.set_stop_loss(symbol, sl, position_side=direction)
                self.refresh_open_orders_cache(symbol)
        except Exception as e:
            logger.error(f"❌ Error en _on_account_update: {e}")

    async def _on_order_trade_update(self, order_data: dict):
        """Manejador de ORDER_TRADE_UPDATE. En simulación no cancela/close"""
        status = order_data.get('X')
        order_type = order_data.get('o')
        symbol = order_data.get('s')
        position_side = order_data.get('ps', 'BOTH')
        logger.info(f"📨 ORDER_UPDATE {symbol} {order_type} {position_side} -> {status}")
        self.refresh_open_orders_cache(symbol)
        if self.simulate:
            return
        if status == "FILLED" and order_type in ['TAKE_PROFIT', 'TAKE_PROFIT_MARKET', 'STOP', 'STOP_MARKET']:
            logger.info(f"🛑 {symbol} - Se llenó {order_type}. Cancelando órdenes residuales y cerrando posición por mercado si sigue abierta.")
            self.api.cancel_all_tp_sl_orders(symbol)
            self.api.cancel_all_limit_orders(symbol)
            self.refresh_open_orders_cache(symbol)
            await asyncio.sleep(1)
            self.refresh_open_orders_cache(symbol)
            pos_info = self.api.get_position_info(symbol)
            if pos_info:
                if isinstance(pos_info, dict):
                    pos_list = [pos_info]
                else:
                    pos_list = pos_info
                for p in pos_list:
                    if abs(float(p.get('positionAmt', 0))) > 0:
                        logger.info(f"⚠️ {symbol} - Cierre de posición por mercado tras TP/SL ejecutado.")
                        self.api.close_all_positions(symbol)
                        break
            with self.open_orders_cache_lock:
                self.open_orders_cache.pop(symbol, None)

    def run(self):
        """Ejecuta el bot en modo paralelo"""
        # Crear loop para WebSocket y ejecutar en hilo separado
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._position_ws(self.listen_key))
        )
        logger.info("🤖 INICIANDO BOT DE TRADING PARALELO")
        logger.info("=" * 60)
        logger.info(f"💰 Balance inicial: ${self.balance:.2f}")
        logger.info(f"📈 Take Profit: {TAKE_PROFIT_PCT}% | Stop Loss: {STOP_LOSS_PCT}%")
        logger.info(f"⏱️ Timeout máximo: {MAX_TRADE_DURATION_MINUTES} minutos")
        logger.info("=" * 60)
        self.running = True
        try:
            logger.info("🔍 Obteniendo símbolos iniciales...")
            self.update_monitored_symbols()
            self.start_price_ws()
            self.start_depth_ws()  # <<< NUEVO

                        # Antes de lanzar hilos de estrategia:
            self.kws = KlineWebSocketCache(
                pairs={sym: ["1m", "5m"] for sym in list(self.monitored_symbols)[:40]},
                max_candles=1500,
                include_open_candle=True
            )
            # self.kws.start()

            self.price_thread = threading.Thread(target=self.price_monitor_thread, daemon=True)
            self.strategy_thread = threading.Thread(target=self.strategy_analysis_thread, daemon=True)
            self._execution_thread = threading.Thread(target=self.execution_thread, daemon=True)
            self.monitor_thread = threading.Thread(target=self.trade_monitor_thread, daemon=True)
            self.price_thread.start()
            self.strategy_thread.start()
            self._execution_thread.start()
            self.monitor_thread.start()
            symbols_update_thread = threading.Thread(target=self.periodic_symbols_update, daemon=True)
            symbols_update_thread.start()
            self.tp_sl_thread = threading.Thread(target=self.tp_sl_watcher, daemon=True)
            self.tp_sl_thread.start()
            # Iniciar hilos para estrategias concurrentes
            # Construir diccionario ticker_sign a partir de las estadísticas de 24 h
            ticker_df = self.get_24h_ticker_stats()
            ticker_sign = {}
            try:
                for _, row in ticker_df.iterrows():
                    sym = row['symbol']
                    change = row['priceChangePercent']
                    ticker_sign[sym.upper()] = change > 0
            except Exception:
                pass
            # Definir modos de inversión
            strategies = [ ('invert_to_long', 'long'),
                           ('invert_to_short', 'short'),
                           ('invert_by_ticker', 'ticker') ]
            for name, mode in strategies:
                t = threading.Thread(target=self._strategy_thread_func, args=(name, ticker_sign, mode), daemon=True)
                self.strategy_threads.append(t)
                t.start()
            scan_counter = 0
            status_counter = 0
            while self.running:
                if scan_counter % 5 == 0:
                    pass
                if status_counter % 10 == 0:
                    self.show_status()
                if status_counter % 20 == 0:
                    self.show_detailed_status()
                scan_counter += 1
                status_counter += 1
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot detenido por el usuario")
            self.running = False
        except Exception as e:
            logger.error(f"❌ Error crítico: {e}")
            self.running = False
        finally:
            self.cleanup()

    def tp_sl_watcher(self):
        """Verifica periódicamente que existan órdenes de TP/SL
        while self.running:
            for symbol in list(self.active_trades.keys()):
                open_orders = self.open_orders_cache.get(symbol, [])
                tp_id = self.tp_sl_order_ids[symbol].get('tp')
                sl_id = self.tp_sl_order_ids[symbol].get('sl')
                open_orders_all = self.open_orders_cache.get(symbol, [])
                open_orders = [o for o in open_orders_all if o.get('symbol', '').upper() == symbol.upper()]
                has_tp = any(o['type'] in ['TAKE_PROFIT', 'TAKE_PROFIT_MARKET'] for o in open_orders)
                has_sl = any(o['type'] in ['STOP', 'STOP_MARKET'] for o in open_orders)
                if not has_tp:
                    # logger.info(f"🔁 Reponiendo TP para {symbol} (no encontrado por ID)")
                    trade = self.active_trades.get(symbol)
                    if trade:
                        if not self.simulate:
                            result_tp = self.api.set_take_profit(symbol, trade.tp, position_side=trade.trade_type)
                            if result_tp and "orderId" in result_tp:
                                self.tp_sl_order_ids[symbol]['tp'] = result_tp['orderId']
                        else:
                            a=1
                           # logger.info(f"🧪 (Simulado) Reposición de TP para {symbol}")
                if not has_sl:
                    #logger.info(f"🔁 Reponiendo SL para {symbol} (no encontrado por ID)")
                    trade = self.active_trades.get(symbol)
                    if trade:
                        if not self.simulate:
                            result_sl = self.api.set_stop_loss(symbol, trade.sl, position_side=trade.trade_type)
                            if result_sl and "orderId" in result_sl:
                                self.tp_sl_order_ids[symbol]['sl'] = result_sl['orderId']
                        else:
                            a=2
                           # logger.info(f"🧪 (Simulado) Reposición de SL para {symbol}")
            time.sleep(10)"""


    def periodic_symbols_update(self):
        """Actualiza símbolos monitoreados periódicamente"""
        while self.running:
            try:
                time.sleep(600)
                logger.info("🔄 Actualizando lista de símbolos...")
                self.update_monitored_symbols()
            except Exception as e:
                logger.error(f"Error actualizando símbolos: {e}")
                time.sleep(300)

    def refresh_open_orders_cache(self, symbol):
        """Refresca el cache de open_orders de forma thread-safe"""
        try:
            open_orders = self.api.get_open_orders(symbol) or []
            with self.open_orders_cache_lock:
                if open_orders:
                    self.open_orders_cache[symbol] = list(open_orders)
                else:
                    self.open_orders_cache.pop(symbol, None)
        except Exception as e:
            logger.error(f"Error refrescando open_orders para {symbol}: {e}")

    def show_status(self):
        """Muestra estado resumido del bot"""
        with self.trades_lock:
            active_trades_copy = dict(self.active_trades)
            active_count = len(active_trades_copy)

        completed_count = len(self.completed_trades)
        if completed_count > 0:
            wins = sum(1 for t in self.completed_trades if t['result'] > 0)
            winrate = (wins / completed_count * 100)
        else:
            winrate = 0

        logger.info(f"\n📊 ESTADO DEL BOT - {datetime.now().strftime('%H:%M:%S')}")
        logger.info("=" * 60)
        logger.info(f"💰 Balance: ${self.balance:.2f}")
        logger.info(f"📈 Operaciones activas: {active_count}")
        logger.info(f"📋 Operaciones completadas: {completed_count}")
        logger.info(f"🔥 Pérdidas consecutivas: {self.consecutive_losses}")
        logger.info(f"🎯 Winrate: {winrate:.1f}%")
        logger.info(f"📡 Símbolos monitoreados: {len(self.monitored_symbols)}")

        if active_trades_copy:
            logger.info("\n🔄 OPERACIONES ACTIVAS:")
            for symbol, trade in active_trades_copy.items():
                duration = datetime.now() - trade.entry_time
                duration_min = int(duration.total_seconds() / 60)
                logger.info(
                    f"   {trade.trade_type} {symbol} | Entrada: ${trade.entry_price:.4f} | "
                    f"Duración: {duration_min}min | Confianza: {trade.confidence:.1f}%"
                )
        logger.info("=" * 60)


    def show_detailed_status(self):
    
        """Muestra estado detallado del bot"""
        logger.info("\n📈 ESTADO DETALLADO DEL BOT")
        logger.info("=" * 60)

        with self.trades_lock:
            active_trades_copy = dict(self.active_trades)

        logger.info(f"💰 Balance actual: ${self.balance:.2f}")
        logger.info(f"📊 Ganancia/Pérdida: ${self.balance - 10000:.2f}")
        logger.info(f"🔥 Pérdidas consecutivas: {self.consecutive_losses}")

        if active_trades_copy:
            logger.info(f"\n🔄 OPERACIONES ACTIVAS ({len(active_trades_copy)}):")
            for symbol, trade in active_trades_copy.items():
                duration = datetime.now() - trade.entry_time
                duration_min = int(duration.total_seconds() / 60)
                current_price = self.data_cache.get_current_price(symbol)
                if current_price:
                    if trade.trade_type == "LONG":
                        unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
                    else:
                        unrealized_pnl = (trade.entry_price - current_price) * trade.quantity
                    logger.info(
                        f"   {trade.trade_type} {symbol} | "
                        f"Entrada: ${trade.entry_price:.4f} | "
                        f"Actual: ${current_price:.4f} | "
                        f"PnL: ${unrealized_pnl:.2f} | "
                        f"Duración: {duration_min}min"
                    )

        if self.completed_trades:
            recent_trades = self.completed_trades[-15:]
            wins = sum(1 for t in self.completed_trades if t['result'] > 0)
            total = len(self.completed_trades)
            winrate = (wins / total * 100) if total > 0 else 0
            total_pnl = sum(t['result'] for t in self.completed_trades)
            logger.info(f"\n📊 ESTADÍSTICAS GENERALES:")
            logger.info(f"   Total operaciones: {total}")
            logger.info(f"   Ganadas: {wins} | Perdidas: {total - wins}")
            logger.info(f"   Winrate: {winrate:.2f}%")
            logger.info(f"   PnL total: ${total_pnl:.2f}")
            logger.info(f"\n📋 ÚLTIMAS 10 OPERACIONES:")
            for trade in recent_trades[-10:]:
                status = "✅" if trade['result'] > 0 else "❌"
                duration = trade['exit_time'] - trade['entry_time']
                duration_min = int(duration.total_seconds() / 60)
                logger.info(
                    f"   {status} {trade['type']} {trade['symbol']} | "
                    f"ROI: {trade['roi']:.2f}% | "
                    f"${trade['result']:.2f} | "
                    f"{duration_min}min | "
                    f"{trade['reason']}"
                )

        cache_symbols = len(self.data_cache.cache_1m)
        symbols_snapshot = list(self.monitored_symbols)  # <— snapshot del set
        fresh_symbols = sum(1 for s in symbols_snapshot if self.data_cache.is_data_fresh(s, 120))
        logger.info(f"\n🗄️ ESTADO DEL CACHE:")
        logger.info(f"   Símbolos en cache: {cache_symbols}")
        logger.info(f"   Datos frescos: {fresh_symbols}/{len(symbols_snapshot)}")
        logger.info(f"\n📡 ESTADO DE COLAS:")
        logger.info(f"   Cola de precios: {self.price_queue.qsize()}")
        logger.info(f"   Cola de señales: {self.signal_queue.qsize()}")
        logger.info(f"   Cola de salidas: {self.exit_queue.qsize()}")
        logger.info("=" * 60)


    def cleanup(self):
        """Limpieza al cerrar el bot"""
        logger.info("🧹 Limpiando recursos...")
        self.running = False
        threads = [self.price_thread, self.strategy_thread, 
                  self._execution_thread, self.monitor_thread]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        # Esperar a que los hilos de estrategia terminen y almacenar resultados
        for t in self.strategy_threads:
            if t and t.is_alive():
                t.join(timeout=5)
        # Cerrar executor
        if self.executor:
            self.executor.shutdown(wait=True)
        # Mostrar resumen final
        self.show_final_summary()
        # Mostrar resultados de estrategias
        if self.strategy_results:
            logger.info("\n📈 RESULTADOS DE ESTRATEGIAS SIMULADAS:")
            for name, metrics in list(self.strategy_results.items()):  # snapshot
                total = metrics['winners'] + metrics['losers']
                accuracy = (metrics['winners'] / total * 100) if total else 0
                profit = metrics['profit']
                loss = metrics['loss']
                logger.info(
                    f"   Estrategia {name} -> Ganadas: {metrics['winners']}, Perdidas: {metrics['losers']}, "
                    f"Precisión: {accuracy:.2f}% | PnL total: {profit + loss:.2f}"
                )
                for bin_key, bin_stats in metrics['bins'].items():
                    bin_total = bin_stats['win'] + bin_stats['loss']
                    bin_acc = (bin_stats['win'] / bin_total * 100) if bin_total else 0
                    logger.info(
                        f"      Confianza {bin_key}: Ganadas {bin_stats['win']}, Perdidas {bin_stats['loss']}, "
                        f"Precisión {bin_acc:.2f}%"
                    )


    def show_final_summary(self):
        """Resumen final del rendimiento del bot"""
        logger.info("\n📊 RESUMEN FINAL")
        logger.info("=" * 60)
        logger.info(f"💰 Balance final: ${self.balance:.2f}")
        logger.info(f"📈 Ganancia/Pérdida: ${self.balance - 10000:.2f}")
        if self.completed_trades:
            total_trades = len(self.completed_trades)
            wins = sum(1 for t in self.completed_trades if t['result'] > 0)
            winrate = (wins / total_trades * 100) if total_trades > 0 else 0
            total_fees = sum(t['fees'] for t in self.completed_trades)
            logger.info(f"📊 Total operaciones: {total_trades}")
            logger.info(f"✅ Ganadas: {wins} | ❌ Perdidas: {total_trades - wins}")
            logger.info(f"🎯 Winrate: {winrate:.2f}%")
            logger.info(f"💸 Total en fees: ${total_fees:.2f}")
            if total_trades > 0:
                avg_trade_duration = sum(
                    (t['exit_time'] - t['entry_time']).total_seconds() / 60 
                    for t in self.completed_trades
                ) / total_trades
                logger.info(f"⏱️ Duración promedio: {avg_trade_duration:.1f} minutos")
        else:
            logger.info("📊 No se completaron operaciones")
        with self.trades_lock:
            if self.active_trades:
                logger.info(f"⚠️ Operaciones activas al cerrar: {len(self.active_trades)}")
                for symbol, trade in self.active_trades.items():
                    logger.info(f"   {trade.trade_type} {symbol} @ ${trade.entry_price:.4f}")
        logger.info("=" * 60)
        logger.info("🤖 Bot finalizado. ¡Gracias por usar el trading bot paralelo!")

    # --- Métodos de estrategia concurrente ---
    def _confidence_bin(self, confidence: float) -> str:
        """
        Determina la etiqueta del rango de confianza para estadísticas.
        Los rangos son: 40-49, 50-59, 60-69, 70-79, 80-89, 90-100.
        """
        if confidence < 40:
            return '<40'
        elif confidence < 50:
            return '40-49'
        elif confidence < 60:
            return '50-59'
        elif confidence < 70:
            return '60-69'
        elif confidence < 80:
            return '70-79'
        elif confidence < 90:
            return '80-89'
        else:
            return '90-100'

    def _strategy_thread_func(self, name: str, ticker_sign: Dict[str, bool], invert_mode: str):
        """
        Función que ejecuta estadísticas en tiempo real para una estrategia dada.

        Args:
            name: nombre de la estrategia (para registrar resultados).
            ticker_sign: diccionario {símbolo -> True si cambio 24h positivo, False si negativo}.
            invert_mode: "long", "short" o "ticker" según el modo de inversión.
        """
        import random
        metrics = {
            'winners': 0,
            'losers': 0,
            'profit': 0.0,
            'loss': 0.0,
            'bins': { '40-49': {'win': 0, 'loss': 0},
                      '50-59': {'win': 0, 'loss': 0},
                      '60-69': {'win': 0, 'loss': 0},
                      '70-79': {'win': 0, 'loss': 0},
                      '80-89': {'win': 0, 'loss': 0},
                      '90-100': {'win': 0, 'loss': 0},
                      '<40': {'win': 0, 'loss': 0} }
        }
        index = 0
        # Procesar señales a medida que se van agregando
        while self.running or index < len(self.signals_log):
            # Mientras haya nuevas señales sin procesar
            while index < len(self.signals_log):
                sig = self.signals_log[index]
                sym = sig['symbol']
                orig_dir = sig['original_signal']
                entry_price = sig['entry_price']
                qty = sig['quantity']
                conf = sig.get('confidence', 0)
                bin_key = self._confidence_bin(conf)
                # Determinar dirección invertida
                if invert_mode == 'long':
                    # 50% de probabilidad de convertir a LONG, de lo contrario mantener original
                    if random.random() < 0.5:
                        new_dir = 'LONG'
                    else:
                        new_dir = orig_dir
                elif invert_mode == 'short':
                    if random.random() < 0.5:
                        new_dir = 'SHORT'
                    else:
                        new_dir = orig_dir
                elif invert_mode == 'ticker':
                    # Convertir según el signo del ticker: si es positivo → LONG, si es negativo → SHORT
                    if ticker_sign.get(sym.upper(), False):
                        new_dir = 'LONG'
                    else:
                        new_dir = 'SHORT'
                else:
                    new_dir = orig_dir
                # Determinar si sería ganadora (TP) o perdedora (SL) comparando con el signo del ticker del símbolo
                tick_pos = ticker_sign.get(sym.upper(), None)
                if tick_pos is None:
                    # Si no tenemos ticker, no podemos evaluar; lo marcamos como perdido
                    metrics['losers'] += 1
                    metrics['bins'][bin_key]['loss'] += 1
                    # PnL como 0
                else:
                    is_correct = (tick_pos and new_dir == 'LONG') or (not tick_pos and new_dir == 'SHORT')
                    # Calcular TP/SL para la nueva dirección
                    # Recalcular niveles (independientemente del original)
                    if new_dir == 'LONG':
                        new_tp = entry_price * (1 + TAKE_PROFIT_PCT / 100)
                        new_sl = entry_price * (1 - STOP_LOSS_PCT / 100)
                        if is_correct:
                            pnl = (new_tp - entry_price) * qty
                            metrics['winners'] += 1
                            metrics['profit'] += pnl
                            metrics['bins'][bin_key]['win'] += 1
                        else:
                            pnl = (new_sl - entry_price) * qty
                            metrics['losers'] += 1
                            metrics['loss'] += pnl
                            metrics['bins'][bin_key]['loss'] += 1
                    else:  # SHORT
                        new_tp = entry_price * (1 - TAKE_PROFIT_PCT / 100)
                        new_sl = entry_price * (1 + STOP_LOSS_PCT / 100)
                        if is_correct:
                            pnl = (entry_price - new_tp) * qty
                            metrics['winners'] += 1
                            metrics['profit'] += pnl
                            metrics['bins'][bin_key]['win'] += 1
                        else:
                            pnl = (entry_price - new_sl) * qty
                            metrics['losers'] += 1
                            metrics['loss'] += pnl
                            metrics['bins'][bin_key]['loss'] += 1
                index += 1
            # Esperar un poco antes de revisar nuevas señales
            time.sleep(1)
        # Guardar métricas
        self.strategy_results[name] = metrics

    def run_signal_simulations(self, sample_fraction: float = 0.5):
        """
        Ejecuta tres simulaciones sobre las señales registradas en `self.signals_log`.

        - Simulación 1: selecciona aleatoriamente un porcentaje de señales (por defecto 50 %) y
          cambia todas las seleccionadas a dirección LONG.
        - Simulación 2: selecciona aleatoriamente el mismo porcentaje de señales y cambia
          todas las seleccionadas a dirección SHORT.
        - Simulación 3: para cada señal, cambia su dirección según el cambio de precio de
          24 h del símbolo: si el cambio es positivo, cambia a LONG; si es negativo,
          cambia a SHORT.

        Para cada simulación, calcula cuántas señales hubieran sido "correctas" si la
        dirección de la señal coincide con el signo del cambio de precio de 24 h del
        símbolo (LONG si cambia > 0 % y SHORT si cambia < 0 %).
        Guarda los resultados en un DataFrame y devuelve ese DataFrame.

        Args:
            sample_fraction: fracción de señales a seleccionar en las simulaciones 1 y 2.

        Returns:
            pandas.DataFrame con los resultados de las simulaciones.
        """
        import random
        # Cargar los cambios de precio de 24 h para todos los símbolos USDT
        ticker_df = self.get_24h_ticker_stats()
        if ticker_df.empty:
            logger.warning("No se pudieron obtener estadísticas de 24 h para los símbolos.\n")
            return pd.DataFrame()
        # Crear un diccionario símbolo -> signo del cambio de 24 h (True si positivo, False si negativo)
        ticker_sign = {}
        for _, row in ticker_df.iterrows():
            symbol = row['symbol']
            change = row['priceChangePercent']
            ticker_sign[symbol.upper()] = change > 0
        # Preparar listas de resultados
        results = []
        # Extraer las señales registradas
        signals = list(self.signals_log)
        total_signals = len(signals)
        if total_signals == 0:
            logger.info("No hay señales registradas para simular.")
            return pd.DataFrame()
        # Obtener las direcciones originales y su corrección según ticker
        original_correct = 0
        for sig in signals:
            sym = sig['symbol']
            orig_dir = sig['original_signal']
            ticker_pos = ticker_sign.get(sym.upper(), None)
            if ticker_pos is None:
                continue
            if (ticker_pos and orig_dir == 'LONG') or (not ticker_pos and orig_dir == 'SHORT'):
                original_correct += 1
        results.append({
            'simulation': 'original',
            'correct_signals': original_correct,
            'total_signals': total_signals,
            'accuracy': original_correct / total_signals
        })
        # Simulación 1: invertir a LONG para sample_fraction de señales
        sim1_correct = 0
        sample_size = int(total_signals * sample_fraction)
        indices = list(range(total_signals))
        random.shuffle(indices)
        selected_indices = set(indices[:sample_size])
        for idx, sig in enumerate(signals):
            sym = sig['symbol']
            ticker_pos = ticker_sign.get(sym.upper(), None)
            if ticker_pos is None:
                continue
            # Dirección invertida: LONG for selected, otherwise original
            if idx in selected_indices:
                inv_dir = 'LONG'
            else:
                inv_dir = sig['original_signal']
            if (ticker_pos and inv_dir == 'LONG') or (not ticker_pos and inv_dir == 'SHORT'):
                sim1_correct += 1
        results.append({
            'simulation': 'invert_to_long',
            'correct_signals': sim1_correct,
            'total_signals': total_signals,
            'accuracy': sim1_correct / total_signals
        })
        # Simulación 2: invertir a SHORT para sample_fraction de señales
        sim2_correct = 0
        random.shuffle(indices)
        selected_indices = set(indices[:sample_size])
        for idx, sig in enumerate(signals):
            sym = sig['symbol']
            ticker_pos = ticker_sign.get(sym.upper(), None)
            if ticker_pos is None:
                continue
            if idx in selected_indices:
                inv_dir = 'SHORT'
            else:
                inv_dir = sig['original_signal']
            if (ticker_pos and inv_dir == 'LONG') or (not ticker_pos and inv_dir == 'SHORT'):
                sim2_correct += 1
        results.append({
            'simulation': 'invert_to_short',
            'correct_signals': sim2_correct,
            'total_signals': total_signals,
            'accuracy': sim2_correct / total_signals
        })
        # Simulación 3: invertir según el ticker de 24 h del símbolo
        sim3_correct = 0
        for sig in signals:
            sym = sig['symbol']
            ticker_pos = ticker_sign.get(sym.upper(), None)
            if ticker_pos is None:
                continue
            inv_dir = 'LONG' if ticker_pos else 'SHORT'
            if (ticker_pos and inv_dir == 'LONG') or (not ticker_pos and inv_dir == 'SHORT'):
                sim3_correct += 1
        results.append({
            'simulation': 'invert_by_ticker',
            'correct_signals': sim3_correct,
            'total_signals': total_signals,
            'accuracy': sim3_correct / total_signals
        })
        # Convertir a DataFrame y devolver
        return pd.DataFrame(results)

# Punto de entrada
def main():
    """Función principal para ejecutar el bot"""
    # Sustituye estas credenciales por las tuyas o deja en blanco si sólo deseas simular
    API_KEY = "j65vqKTAEvJtOZMCQbSiH5GZXfzyg1W70dWvhnb5DHxMOlLaW1JlrohJtYf8hJMH"
    API_SECRET = "qBqVSu0b0stLoN5hWEo5TAeK0IyfI4bNP1kQh7X3JoXVlzBOVutMSr0CWtvTua0O"
    # Crear el bot en modo simulación (por defecto simulate=True)
    bot = ParallelTradingBot(api_key=API_KEY, api_secret=API_SECRET, testnet=False, simulate=True)
    try:
        bot.run()
    except Exception as e:
        logger.error(f"Error ejecutando bot: {e}")
    finally:
        # Al finalizar, ejecutar simulaciones de señales si hay señales registradas
        results_df = bot.run_signal_simulations()
        if not results_df.empty:
            # Guardar resultados en un CSV para análisis futuro
            results_csv_path = 'simulation_results.csv'
            results_df.to_csv(results_csv_path, index=False)
            logger.info(f"Resultados de simulación guardados en {results_csv_path}")
            # Sincronizar el archivo para que esté disponible al usuario
            # Nota: la función computer.sync_file no se puede llamar desde aquí
        bot.cleanup()

if __name__ == "__main__":
    main()

