import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
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
import os
import tempfile
from pathlib import Path



# Import the corrected Binance API wrapper
from binance_api_mejorado import BinanceAPI

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Trading parameters (same as original strategy)
TAKE_PROFIT_PCT = 6.6
STOP_LOSS_PCT = 1.4
MAX_TRADE_DURATION_MINUTES = 30
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
    def __init__(self, max_symbols: int = 40, max_candles: int = 1500):
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
    - (Nuevo) Backfill por REST al iniciar para cada (symbol, interval)
    - Suscribe a <symbol>@kline_<interval> para actualizar en vivo
    - Cache thread-safe por (symbol, interval) con máx N velas
    """
    BASE_WS_URL = "wss://fstream.binance.com/ws"
    BASE_REST_URL = "https://fapi.binance.com"

    def __init__(self,
                 pairs: Dict[str, List[str]],
                 max_candles: int = 1500,
                 include_open_candle: bool = True,
                 backfill_on_start: bool = True,
                 rest_limits: Optional[Dict[str, int]] = None,
                 rest_timeout: float = 6.0,
                 rest_min_sleep: float = 0.12,
                 session: Optional[requests.Session] = None):
        """
        pairs: {'BTCUSDT': ['1m','5m'], 'ETHUSDT': ['1m']}
        max_candles: máximo de velas por (symbol, interval)
        include_open_candle: si True, la vela abierta (x=False) se mantiene/actualiza
        backfill_on_start: si True, descarga histórico por REST al iniciar
        rest_limits: dict opcional {interval -> limit} (p. ej. {'1m': 720, '5m': 300})
        rest_timeout: timeout para las llamadas REST
        rest_min_sleep: pausa mínima entre requests para evitar rate-limit
        session: requests.Session reutilizable (opcional)
        """
        self.pairs = {s.upper(): ([i] if isinstance(i, str) else [x for x in i])
                      for s, i in pairs.items()}
        self.max_candles = int(max_candles)
        self.include_open = bool(include_open_candle)

        self.backfill_on_start = bool(backfill_on_start)
        self.rest_limits = rest_limits or {}
        self.rest_timeout = float(rest_timeout)
        self.rest_min_sleep = float(rest_min_sleep)
        self._rest = session or requests.Session()

        # buffers[(symbol, interval)] -> deque de dicts con campos de kline
        self.buffers: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=self.max_candles))
        self.lock = threading.Lock()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._tasks = {}
        self._thread: Optional[threading.Thread] = None

    # -----------------------
    # Backfill por REST (nuevo)
    # -----------------------
    def _backfill_symbol_interval(self, symbol: str, interval: str):
        """Descarga histórico por REST para (symbol, interval) y lo vuelca al buffer."""
        sym = symbol.upper()
        itv = interval
        # Limite de velas por intervalo (cap a max_candles)
        limit_default = min(720, self.max_candles)  # sensato para 1m
        limit = int(min(self.rest_limits.get(itv, limit_default), self.max_candles))
        if limit <= 0:
            return

        url = f"{self.BASE_REST_URL}/fapi/v1/klines"
        params = {"symbol": sym, "interval": itv, "limit": limit}
        try:
            resp = self._rest.get(url, params=params, timeout=self.rest_timeout)
            resp.raise_for_status()
            data = resp.json()  # lista de klines
        except Exception as e:
            print(f"🔴 Backfill REST falló {sym} {itv}: {e}")
            return

        # Parsear a nuestro formato interno (igual al de WS)
        rows = []
        for k in data:
            # Binance kline:
            # [0] openTime, [1] open, [2] high, [3] low, [4] close, [5] volume,
            # [6] closeTime, [7] quoteVol, [8] trades, [9] takerBuyVol, [10] takerBuyQuoteVol, [11] ignore
            try:
                row = {
                    "open_time": int(k[0]),
                    "close_time": int(k[6]),
                    "symbol": sym,
                    "interval": itv,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "trades": int(k[8]),
                    "quote_volume": float(k[7]),
                    "taker_buy_volume": float(k[9]),
                    "taker_buy_quote_volume": float(k[10]),
                    "is_closed": True,   # histórico por REST siempre cerrado
                }
                rows.append(row)
            except Exception:
                # Si una fila viene corrupta, la saltamos
                continue

        if not rows:
            return

        # Volcar en buffer con lock; evitar duplicados por open_time
        key = (sym, itv)
        with self.lock:
            buf = self.buffers[key]
            existing_otimes = {r["open_time"] for r in buf} if len(buf) else set()
            for r in rows:
                if r["open_time"] in existing_otimes:
                    # si ya existe esa vela, actualizamos la última ocurrencia
                    # (histórico no debería duplicarse, pero por seguridad)
                    # búsqueda simple desde el final
                    for idx in range(len(buf) - 1, -1, -1):
                        if buf[idx]["open_time"] == r["open_time"]:
                            buf[idx] = r
                            break
                else:
                    buf.append(r)
                    existing_otimes.add(r["open_time"])

    def _backfill_all(self):
        """Backfill por REST para todos los (symbol, interval)."""
        # Secuencial y con pequeña pausa para evitar -1003
        for symbol, intervals in self.pairs.items():
            for interval in intervals:
                self._backfill_symbol_interval(symbol, interval)
                # pequeña pausa entre requests
                time.sleep(self.rest_min_sleep)

    # -----------------------
    # WebSocket
    # -----------------------
    async def _run_stream(self, symbol: str, interval: str):
        stream = f"{symbol.lower()}@kline_{interval}"
        url = f"{self.BASE_WS_URL}/{stream}"
        backoff = 1
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=None) as ws:
                    print(f"🟢 Kline WS conectado: {stream}")
                    backoff = 1
                    while self._running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        if data.get("e") != "kline":
                            continue
                        k = data.get("k", {})
                        is_closed = bool(k.get("x", False))
                        if (not is_closed) and (not self.include_open):
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
                            buf = self.buffers[key]
                            if len(buf) > 0 and buf[-1]["open_time"] == row["open_time"]:
                                buf[-1] = row
                            else:
                                # Si WS envía una vela cerrada más vieja que el final, hacemos un merge simple
                                if buf and row["open_time"] < buf[-1]["open_time"]:
                                    # Inserción ordenada (rara vez ocurre). Linear, pero buffers son pequeños.
                                    inserted = False
                                    for i in range(len(buf)):
                                        if row["open_time"] < buf[i]["open_time"]:
                                            buf.insert(i, row)
                                            inserted = True
                                            break
                                    if not inserted:
                                        buf.append(row)
                                else:
                                    buf.append(row)
            except Exception as e:
                print(f"🔴 Error WS {stream}: {e}. Reintentando en {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    # -----------------------
    # Ciclo de vida
    # -----------------------
    def start(self):
        if self._running:
            return
        self._running = True

        # 1) Backfill por REST (opcional)
        if self.backfill_on_start:
            print("📥 Backfill REST inicial de klines…")
            self._backfill_all()

        # 2) Iniciar loop y tareas WS
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        for symbol, intervals in self.pairs.items():
            for interval in intervals:
                coro = self._run_stream(symbol, interval)
                task = asyncio.run_coroutine_threadsafe(coro, self._loop)
                self._tasks[(symbol, interval)] = task
        print("🚀 KlineWebSocketCache iniciado (REST backfill + WS en vivo)")

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

    # -----------------------
    # Lecturas
    # -----------------------
    def get_dataframe(self, symbol: str, interval: str, only_closed: bool = False) -> pd.DataFrame:
        """
        Devuelve un DataFrame con las velas acumuladas (REST backfill + WS).
        Si only_closed=True, filtra por is_closed=True.
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
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        return df[[
            "timestamp","open","high","low","close","volume",
            "close_time","trades","quote_volume","taker_buy_volume","taker_buy_quote_volume","is_closed"
        ]].reset_index(drop=True)

    def get_last_closed(self, symbol: str, interval: str) -> Optional[dict]:
        """Devuelve la última vela cerrada disponible."""
        df = self.get_dataframe(symbol, interval, only_closed=True)
        if df.empty:
            return None
        return df.iloc[-1].to_dict()


from typing import Literal

# ====== Cadena de Markov 1D por símbolo (estados: DOWN, FLAT, UP) ======
class MarkovChain1D:
    STATES = ("DOWN", "FLAT", "UP")
    IDX = {"DOWN": 0, "FLAT": 1, "UP": 2}

    def __init__(self, up_thr: float = 0.014, dn_thr: Optional[float] = None, alpha: float = 1.0):
        """
        up_thr: umbral de retorno para etiquetar UP (p.ej. 0.1%)
        dn_thr: umbral (negativo) para etiquetar DOWN; si None = -up_thr
        alpha : suavizado tipo Dirichlet (Laplace) para probabilidades
        """
        self.up_thr = float(up_thr)
        self.dn_thr = float(dn_thr if dn_thr is not None else -up_thr)
        self.alpha = float(alpha)
        self.C = np.zeros((3, 3), dtype=np.float64)  # matriz de conteos de transiciones
        self.last_state: Optional[str] = None
        self.n_obs = 0

    def _label(self, ret: float) -> str:
        if ret >= self.up_thr:  return "UP"
        if ret <= self.dn_thr:  return "DOWN"
        return "FLAT"

    def update_from_return(self, ret: float) -> str:
        s = self._label(float(ret))
        if self.last_state is not None:
            i = self.IDX[self.last_state]; j = self.IDX[s]
            self.C[i, j] += 1.0
        self.last_state = s
        self.n_obs += 1
        return s

    def row_probs(self, state: Optional[str] = None) -> np.ndarray:
        """Prob(next | state) con suavizado alpha."""
        if state is None:
            # distribución marginal aproximada
            M = self.C + self.alpha
            return (M.sum(axis=0) / M.sum()).astype(np.float64)
        i = self.IDX[state]
        row = self.C[i, :] + self.alpha
        return (row / row.sum()).astype(np.float64)

    def next_probs(self) -> np.ndarray:
        """Probabilidades desde el último estado conocido (o uniforme si None)."""
        if self.last_state is None:
            return np.ones(3, dtype=np.float64) / 3.0
        return self.row_probs(self.last_state)

    def predict(self) -> Tuple[str, float, float]:
        """Devuelve (dir_pred, p_up, p_down)."""
        p = self.next_probs()
        p_down, p_flat, p_up = float(p[0]), float(p[1]), float(p[2])
        if p_up > max(p_down, p_flat):   return "UP", p_up, p_down
        if p_down > max(p_up, p_flat):   return "DOWN", p_up, p_down
        return "FLAT", p_up, p_down
    
    def save_markov(self, path: str = "markov_state.json"):
        try:
            payload = {}
            for sym, mc in self.mc_models.items():
                payload[sym] = {
                    "up_thr": mc.up_thr, "dn_thr": mc.dn_thr, "alpha": mc.alpha,
                    "C": mc.C.tolist(), "last_state": mc.last_state, "n_obs": mc.n_obs
                }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception as e:
            logger.error(f"No se pudo guardar Markov: {e}")

    def load_markov(self, path: str = "markov_state.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.mc_models = {}
            for sym, obj in payload.items():
                mc = MarkovChain1D(up_thr=obj["up_thr"], dn_thr=obj["dn_thr"], alpha=obj["alpha"])
                mc.C = np.array(obj["C"], dtype=np.float64)
                mc.last_state = obj["last_state"]
                mc.n_obs = int(obj["n_obs"])
                self.mc_models[sym] = mc
        except Exception:
            pass



# ====== Gestor de riesgo: TP/SL dinámicos y sizing por riesgo ======
class RiskManager:
    def __init__(self,
                 base_tp_pct: float = 6.6,
                 base_sl_pct: float = 1.4,
                 min_tp_pct: float = 0.3,
                 min_sl_pct: float = 0.3,
                 max_tp_pct: float = 6.6,
                 max_sl_pct: float = 2.0,
                 target_risk_usdt: float = 2.5,
                 min_notional_usdt: float = 5.5,
                 max_notional_usdt: float = 15.0):
        self.base_tp = float(base_tp_pct)
        self.base_sl = float(base_sl_pct)
        self.min_tp = float(min_tp_pct)
        self.min_sl = float(min_sl_pct)
        self.max_tp = float(max_tp_pct)
        self.max_sl = float(max_sl_pct)
        self.target_risk = float(target_risk_usdt)
        self.min_notional = float(min_notional_usdt)
        self.max_notional = float(max_notional_usdt)

    @staticmethod
    def _clip(x, lo, hi): return float(max(lo, min(hi, x)))

    def dynamic_tp_sl_pct(self,
                          atr_frac: float,
                          vol_score: float,
                          confidence: float,
                          mc_delta: float) -> Tuple[float, float]:
        """
        Calcula TP/SL % dinámicos:
        - atr_frac: ATR% (e.g. 0.004 => 0.4%)
        - vol_score: mezcla std anualizada y ATR%
        - confidence: 50..98
        - mc_delta: p_up - p_down en [-1..1]
        """
        atr_pct = float(max(0.0001, atr_frac)) * 100.0  # en %
        conf01 = self._clip((confidence - 50.0) / 48.0, 0.0, 1.0)   # 0..1
        mc01   = self._clip((abs(mc_delta)), 0.0, 1.0)              # 0..1

        # Pisos por volatilidad reciente
        floor_tp = max(self.min_tp, 1.0 * atr_pct)
        floor_sl = max(self.min_sl, 0.7 * atr_pct)

        # Ajustes por confianza y sesgo Markov
        tp = self.base_tp * (1.10 + 0.45*conf01 + 0.35*mc01)    # ↑ con confianza
        sl = self.base_sl * (0.75 - 0.60*conf01 - 0.30*mc01)   # ↓ con confianza

        tp = self._clip(tp, floor_tp, self.max_tp)
        sl = self._clip(sl, floor_sl, self.max_sl)
        return tp, sl

    def position_size_by_risk(self, price: float, sl_pct: float) -> float:
        """
        Sizing por riesgo aproximado: riesgo ≈ price * sl_pct% * qty.
        """
        risk_per_unit = price * (sl_pct/100.0)
        if risk_per_unit <= 0:
            return self.min_notional / price

        qty_risk = self.target_risk / risk_per_unit
        qty_min  = self.min_notional / price
        qty_max  = self.max_notional / price
        return float(self._clip(qty_risk, qty_min, qty_max))


class PersistenceManager:
    """
    Guarda/carga estado de aprendizaje del bot con escritura atómica y versionado.
    Contiene:
      - Markov por símbolo: C, last_state, n_obs, umbrales
      - Modelo ML online por símbolo: w, b, mu, sigma, h, thr, n
      - Parámetros del RiskManager
    Opcional: puedes extender a señales/trades si quieres.
    """
    SCHEMA_VERSION = 1

    def __init__(self, path: Path):
        self.path = Path(path)

    @staticmethod
    def _atomic_write(path: Path, data: str):
        path = Path(path)
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp_name, path)  # atómico en la mayoría de OS
        except Exception:
            try:
                os.remove(tmp_name)
            except Exception:
                pass
            raise

    def dump_from_bot(self, bot: "ParallelTradingBot") -> dict:
        # Empaquetar con locks
        with bot.persist_lock:
            # Markov
            mc_payload = {}
            for sym, mc in bot.mc_models.items():
                mc_payload[sym] = {
                    "up_thr": mc.up_thr,
                    "dn_thr": mc.dn_thr,
                    "alpha": mc.alpha,
                    "C": mc.C.tolist(),
                    "last_state": mc.last_state,
                    "n_obs": mc.n_obs,
                }

            # ML online por símbolo
            ml_payload = {}
            for sym, st in bot.ml_state.items():
                model = st.get("model")
                if model is None:
                    continue
                ml_payload[sym] = {
                    "w": model.w.tolist(),
                    "b": float(model.b),
                    "mu": st["mu"].tolist(),
                    "sigma": st["sigma"].tolist(),
                    "h": int(st.get("h", 5)),
                    "thr": float(st.get("thr", 0.001)),
                    "n": int(st.get("n", 0)),
                }

            # Riesgo
            rm = bot.risk_mgr
            risk_payload = {
                "base_tp": rm.base_tp,
                "base_sl": rm.base_sl,
                "min_tp": rm.min_tp,
                "min_sl": rm.min_sl,
                "max_tp": rm.max_tp,
                "max_sl": rm.max_sl,
                "target_risk": rm.target_risk,
                "min_notional": rm.min_notional,
                "max_notional": rm.max_notional,
            }

            return {
                "schema": self.SCHEMA_VERSION,
                "mc_models": mc_payload,
                "ml_state": ml_payload,
                "risk_manager": risk_payload,
                "timestamp": datetime.now().isoformat(),
            }

    def load_into_bot(self, bot: "ParallelTradingBot", payload: dict):
        if not payload or not isinstance(payload, dict):
            return
        # En el futuro: migraciones por versión
        _ = int(payload.get("schema", 1))

        with bot.persist_lock:
            # --- Markov ---
            mc_payload = payload.get("mc_models", {})
            bot.mc_models = {}
            for sym, obj in mc_payload.items():
                mc = MarkovChain1D(
                    up_thr=float(obj.get("up_thr", getattr(bot, "BTC_MOM_THR", 0.001))),
                    dn_thr=float(obj.get("dn_thr", -getattr(bot, "BTC_MOM_THR", 0.001))),
                    alpha=float(obj.get("alpha", 1.0)),
                )
                C = np.array(obj.get("C", [[0,0,0],[0,0,0],[0,0,0]]), dtype=np.float64)
                if C.shape == (3,3):
                    mc.C = C
                mc.last_state = obj.get("last_state")
                mc.n_obs = int(obj.get("n_obs", 0))
                bot.mc_models[sym] = mc

            # --- ML online ---
            ml_payload = payload.get("ml_state", {})
            bot.ml_state = {}
            for sym, st in ml_payload.items():
                mu = np.array(st["mu"], dtype=np.float64)
                sigma = np.array(st["sigma"], dtype=np.float64)
                d = len(mu)
                model = OnlineLogisticRegression(n_features=d, lr=0.05, l2=1e-4)
                model.w = np.array(st["w"], dtype=np.float64)
                model.b = float(st["b"])
                bot.ml_state[sym] = {
                    "model": model,
                    "mu": mu,
                    "sigma": sigma,
                    "h": int(st.get("h", 5)),
                    "thr": float(st.get("thr", 0.001)),
                    "n": int(st.get("n", 0)),
                }

            # --- Riesgo ---
            risk = payload.get("risk_manager", {})
            bot.risk_mgr.base_tp = float(risk.get("base_tp", bot.risk_mgr.base_tp))
            bot.risk_mgr.base_sl = float(risk.get("base_sl", bot.risk_mgr.base_sl))
            bot.risk_mgr.min_tp = float(risk.get("min_tp", bot.risk_mgr.min_tp))
            bot.risk_mgr.min_sl = float(risk.get("min_sl", bot.risk_mgr.min_sl))
            bot.risk_mgr.max_tp = float(risk.get("max_tp", bot.risk_mgr.max_tp))
            bot.risk_mgr.max_sl = float(risk.get("max_sl", bot.risk_mgr.max_sl))
            bot.risk_mgr.target_risk = float(risk.get("target_risk", bot.risk_mgr.target_risk))
            bot.risk_mgr.min_notional = float(risk.get("min_notional", bot.risk_mgr.min_notional))
            bot.risk_mgr.max_notional = float(risk.get("max_notional", bot.risk_mgr.max_notional))

    def save(self, bot: "ParallelTradingBot"):
        payload = self.dump_from_bot(bot)
        data = json.dumps(payload, ensure_ascii=False, indent=2)
        self._atomic_write(self.path, data)

    def load(self, bot: "ParallelTradingBot"):
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.load_into_bot(bot, payload)
        except Exception as e:
            print(f"⚠️ No se pudo cargar estado desde {self.path}: {e}")




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

        self.symbol_filters = {}          # symbol -> {'stepSize','tickSize','minQty','minNotional'}
        self.leverage_cache = {}          # symbol -> {'leverage': int, 'ts': datetime}
        self.rest_circuit_until = 0.0     # epoch seconds: si hay -1003, cortacircuito temporal
        self.rest_sema = threading.Semaphore(5)  # no más de 5 REST simultáneos desde esta clase

                # === NUEVO: estados por símbolo ===
        self.mc_models: Dict[str, MarkovChain1D] = {}   # Cadena de Markov por símbolo
        self.risk_mgr = RiskManager(
            base_tp_pct=TAKE_PROFIT_PCT,
            base_sl_pct=STOP_LOSS_PCT,
            target_risk_usdt=1,     # ajusta si quieres más/menos riesgo por trade
            min_notional_usdt=5.5,
            max_notional_usdt=15.0
        )


                # === Persistencia ===
        self.persist_lock = threading.Lock()
        self.persist_path = Path("bot_state.json")           # cámbialo si quieres
        self.autosave_secs = 300                              # cada 5 min
        self._autosave_thread = None
        self._autosave_running = False






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


    # ====== MARKOV: helpers por símbolo ======
    def _get_mc(self, symbol: str) -> MarkovChain1D:
        mc = self.mc_models.get(symbol)
        if mc is None:
            # Usa tu mismo umbral de momentum 3m como separador de estados
            mc = MarkovChain1D(up_thr=self.BTC_MOM_THR, alpha=1.0)
            self.mc_models[symbol] = mc
        return mc

    def _update_markov_from_df(self, symbol: str, df_1m: pd.DataFrame, win:int=3) -> Tuple[str, float, float, float]:
        """
        Actualiza MC con retorno t->t-win y devuelve: (mc_dir, p_up, p_down, mc_delta)
        """
        mc = self._get_mc(symbol)
        if df_1m is None or df_1m.empty or len(df_1m) < win+1:
            # sin datos suficientes, sólo leer predicción actual
            dirp, p_up, p_dn = mc.predict()
            return dirp, p_up, p_dn, (p_up - p_dn)

        ret = (df_1m['close'].iloc[-1] - df_1m['close'].iloc[-win]) / df_1m['close'].iloc[-win]
        mc.update_from_return(float(ret))
        dirp, p_up, p_dn = mc.predict()
        return dirp, p_up, p_dn, (p_up - p_dn)

    def _update_markov_from_trade(self, symbol: str, entry_price: float, exit_price: float):
        """Refuerzo extra: actualiza MC con retorno del trade completo."""
        if entry_price is None or exit_price is None or entry_price <= 0:
            return
        ret = (exit_price - entry_price) / entry_price
        self._get_mc(symbol).update_from_return(float(ret))


    # ====== Persistencia: hooks de alto nivel ======
    def _persistence_manager(self) -> PersistenceManager:
        return PersistenceManager(self.persist_path)

    def load_state(self):
        try:
            pm = self._persistence_manager()
            pm.load(self)
            logger.info(f"💾 Estado cargado desde {self.persist_path}")
        except Exception as e:
            logger.warning(f"No se pudo cargar estado: {e}")

    def save_state(self):
        try:
            pm = self._persistence_manager()
            pm.save(self)
            logger.info(f"💾 Estado guardado en {self.persist_path}")
        except Exception as e:
            logger.error(f"No se pudo guardar estado: {e}")

    # ====== Autosave periódico ======
    def _autosave_loop(self):
        logger.info(f"🧷 Autosave cada {self.autosave_secs}s habilitado")
        while self._autosave_running:
            try:
                for _ in range(self.autosave_secs):
                    if not self._autosave_running:
                        break
                    time.sleep(1)
                if not self._autosave_running:
                    break
                self.save_state()
            except Exception as e:
                logger.error(f"Autosave falló: {e}")

    def start_autosave(self):
        if self._autosave_running:
            return
        self._autosave_running = True
        self._autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True)
        self._autosave_thread.start()

    def stop_autosave(self):
        self._autosave_running = False
        if self._autosave_thread and self._autosave_thread.is_alive():
            self._autosave_thread.join(timeout=5)



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
        df['EMA9'] = df['close'].ewm(span=25, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=16, adjust=False).mean()
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
            """
            Señal combinada (RSI + Logística + Markov) SIN REST:
            - Datos de KWS/DataCache
            - TP/SL dinámicos por riesgo/volatilidad/confianza
            """
            try:
                # ---------- 1) Obtener datos 1m/5m SIN REST ----------
                df_1m, df_5m = self.data_cache.get_data(symbol)

                def _df_from_kws(sym: str, itv: str) -> pd.DataFrame:
                    if not hasattr(self, "kws") or self.kws is None:
                        return pd.DataFrame()
                    df = self.kws.get_dataframe(sym, itv, only_closed=False)
                    if df.empty: return df
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy().reset_index(drop=True)

                need_1m = (df_1m is None) or df_1m.empty or (len(df_1m) < 60)
                need_5m = (df_5m is None) or df_5m.empty or (len(df_5m) < 10)
                if need_1m or need_5m:
                    df1_ws = _df_from_kws(symbol, "1m") if need_1m else df_1m
                    df5_ws = _df_from_kws(symbol, "5m") if need_5m else df_5m
                    if df1_ws is not None and not df1_ws.empty and df5_ws is not None and not df5_ws.empty:
                        self.data_cache.update_data(symbol, df1_ws, df5_ws)
                        df_1m, df_5m = df1_ws, df5_ws

                if df_1m is None or df_1m.empty or len(df_1m) < 60:
                    return {}
                if df_5m is None:
                    df_5m = pd.DataFrame()

                # ---------- 2) Indicadores base ----------
                df_1m = self.calculate_indicators(df_1m.copy())
                if not df_5m.empty:
                    df_5m = self.calculate_indicators(df_5m.copy())

                # ---------- 3) ML por símbolo ----------
                horizon = 5
                thr = 0.001
                state, ready = self._ml_get_or_train(symbol, df_1m, horizon=horizon, thr=thr)
                if not ready:
                    return {}
                feats_df = self._ml_build_features(df_1m)
                if feats_df.empty:
                    return {}
                x_raw = feats_df.drop(columns=['close','timestamp']).iloc[-1].values.astype(np.float64)
                x = self._std_scale(x_raw, state['mu'], state['sigma'])
                p_up = float(state['model'].predict_proba(x))

                # ---------- 4) Volatilidad / ATR% ----------
                vol_metrics = self.calculate_volatility_metrics(df_1m) or {}
                vol_score = float(vol_metrics.get('volatility_score', 0.0) or 0.0)
                high_vol = vol_score > 0.02
                atr_frac_last = float(feats_df['atr_frac'].iloc[-1]) if 'atr_frac' in feats_df.columns else 0.003

                # Tendencias
                up_1m = down_1m = False
                if 'EMA9' in df_1m.columns and 'EMA26' in df_1m.columns:
                    ema_fast_1m = df_1m['EMA9'].iloc[-1]
                    ema_slow_1m = df_1m['EMA26'].iloc[-1]
                    up_1m = bool(ema_fast_1m > ema_slow_1m)
                    down_1m = bool(ema_fast_1m < ema_slow_1m)
                else:
                    ema_fast_1m = ema_slow_1m = float('nan')

                up_5m = down_5m = False
                if not df_5m.empty and ('EMA9' in df_5m.columns) and ('EMA26' in df_5m.columns):
                    ema9_5m = df_5m['EMA9'].iloc[-1]
                    ema26_5m = df_5m['EMA26'].iloc[-1]
                    up_5m = (not np.isnan(ema9_5m)) and (not np.isnan(ema26_5m)) and (ema9_5m > ema26_5m)
                    down_5m = (not np.isnan(ema9_5m)) and (not np.isnan(ema26_5m)) and (ema9_5m < ema26_5m)

                # ---------- 5) RSI por regresión ----------
                rsi_dir = None; rsi_strength = "none"; rsi_last = float('nan')
                if 'RSI' in df_1m.columns:
                    rsi_last = float(df_1m['RSI'].iloc[-1])
                    rsi_win = self.RSI_SLOPE_WIN
                    rsi_series = df_1m['RSI'].tail(rsi_win).dropna()
                    if len(rsi_series) >= max(8, rsi_win // 2):
                        rsi_slope, rsi_r2 = self._linreg_slope_r2(rsi_series.values)
                        if abs(rsi_slope) >= self.RSI_SLOPE_THR and rsi_r2 >= self.RSI_TREND_MIN_R2:
                            rsi_dir = "LONG" if rsi_slope > 0 else "SHORT"; rsi_strength = "strong"
                        elif abs(rsi_slope) >= 1e-3:
                            rsi_dir = "LONG" if rsi_slope > 0 else "SHORT"; rsi_strength = "weak"
                    else:
                        rsi_slope, rsi_r2 = 0.0, 0.0
                else:
                    rsi_slope, rsi_r2 = 0.0, 0.0

                if high_vol and not np.isnan(rsi_last) and (rsi_last >= self.RSI_HARD_OVERBOUGHT or rsi_last <= self.RSI_HARD_OVERSOLD):
                    if self.RSI_EXTREME_MODE == 'block':
                        return {}
                    else:
                        rsi_dir = "SHORT" if rsi_last >= self.RSI_HARD_OVERBOUGHT else "LONG"

                # ---------- 6) Lógica ML pura ----------
                long_th  = 0.60 if high_vol else 0.65
                short_th = 0.40 if high_vol else 0.35
                logi_dir = None
                if (p_up >= long_th) and high_vol and (not down_5m) and up_1m:
                    logi_dir = "LONG"
                elif (p_up <= short_th) and high_vol and (not up_5m) and down_1m:
                    logi_dir = "SHORT"

                # ---------- 7) Markov por símbolo ----------
                mc_dir, mc_p_up, mc_p_dn, mc_delta = self._update_markov_from_df(symbol, df_1m, win=3)

                # ---------- 8) Fusión de señales ----------
                # Base: prioriza RSI si fue marcado como primario; sino ML; después Markov como desempate.
                signal_type = None
                if self.RSI_TREND_PRIMARY and rsi_dir is not None:
                    signal_type = rsi_dir
                elif logi_dir is not None:
                    signal_type = logi_dir
                elif rsi_dir is not None:
                    signal_type = rsi_dir

                # Si aún no hay señal, deja que Markov defina cuando es muy contundente
                if signal_type is None and abs(mc_delta) >= 0.25 and high_vol:
                    signal_type = "LONG" if mc_dir == "UP" else ("SHORT" if mc_dir == "DOWN" else None)

                if signal_type is None:
                    return {}

                # Si Markov contradice fuertemente y es alta vol, invierte (override suave)
                if abs(mc_delta) >= 0.45 and high_vol:
                    if mc_dir == "UP" and signal_type == "SHORT":
                        signal_type = "LONG"
                    elif mc_dir == "DOWN" and signal_type == "LONG":
                        signal_type = "SHORT"

                # ---------- 9) Confianza combinada ----------
                # Confianza RSI
                conf_rsi = 55.0
                conf_rsi += min(20.0, abs(rsi_slope) * 40.0)
                conf_rsi += min(15.0, rsi_r2 * 30.0)
                if (rsi_dir is not None) and (rsi_strength == "weak"):
                    conf_rsi -= 8.0
                conf_rsi = max(40.0, conf_rsi)
                if (rsi_dir is not None) and (logi_dir is not None) and (rsi_dir == logi_dir):
                    conf_rsi += 5.0

                # Confianza ML
                ema_gap = self._safe_div(abs(ema_fast_1m - ema_slow_1m), abs(ema_slow_1m))
                gap = abs(p_up - 0.5) * 2.0
                conf_ml = 50 + min(25, gap * 40 * (1.0 if high_vol else 0.7)) + min(20, ema_gap * 2000) + min(10, vol_score * 300)
                conf_ml = float(np.clip(conf_ml, 50, 98))

                # Confianza Markov (fuerte si hay datos)
                mc_strength = abs(mc_delta) * min(1.0, self._get_mc(symbol).n_obs / 150.0)  # 0..1
                conf_mc = 50 + 40*mc_strength

                # Fusión ponderada (RSI primario si así está configurado)
                if self.RSI_TREND_PRIMARY and rsi_dir is not None:
                    confidence = 0.50*conf_rsi + 0.35*conf_ml + 0.15*conf_mc
                else:
                    confidence = 0.40*conf_ml + 0.35*conf_rsi + 0.25*conf_mc
                confidence = float(np.clip(confidence, 50, 98))

                # ---------- 10) Precio actual ----------
                current_price = self.get_ws_price(symbol)
                if current_price is None:
                    current_price = float(df_1m['close'].iloc[-1])

                # (tu estrategia actual invierte al final)
                signal_type = 'LONG' if signal_type == 'SHORT' else 'SHORT'    

                # ---------- 11) TP/SL dinámicos por riesgo ----------
                tp_pct, sl_pct = self.risk_mgr.dynamic_tp_sl_pct(
                    atr_frac=atr_frac_last,
                    vol_score=vol_score,
                    confidence=confidence,
                    mc_delta=mc_delta
                )
                if signal_type == "LONG":
                    tp = current_price * (1 + tp_pct/100.0)
                    sl = current_price * (1 - sl_pct/100.0)
                else:
                    tp = current_price * (1 - tp_pct/100.0)
                    sl = current_price * (1 + sl_pct/100.0)

                

                rsi_1m = float(df_1m['RSI'].iloc[-1]) if 'RSI' in df_1m.columns else float('nan')
                return {
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'current_price': float(current_price),
                    'confidence': float(confidence),
                    'volatility_score': float(vol_score),
                    'p_up': float(p_up),
                    'ema_fast': float(ema_fast_1m),
                    'rsi': rsi_1m,
                    'ema_slow': float(ema_slow_1m),
                    # NUEVO: info Markov y riesgo
                    'mc_dir': mc_dir, 'mc_p_up': float(mc_p_up), 'mc_p_down': float(mc_p_dn), 'mc_delta': float(mc_delta),
                    'atr_frac': float(atr_frac_last),
                    'tp': float(tp), 'sl': float(sl), 'tp_pct': float(tp_pct), 'sl_pct': float(sl_pct)
                }
            except Exception as e:
                logger.debug(f"Error analizando (ML+MC) {symbol} [sin REST]: {e}")
                return {}

   
    def calculate_position_size(self, current_price: float, dynamic_sl_pct: Optional[float] = None) -> float:
        """
        Sizing:
        - Si hay SL% dinámico: usa gestor de riesgo (riesgo fijo en USDT).
        - Si no, cae al sizing notional original.
        """
        base_notional = 5.5
        qty = base_notional / current_price
        min_notional = 5.5
        if qty * current_price < min_notional:
            qty = min_notional / current_price
       
        return float(qty)


    def price_monitor_thread(self):
        logger.info("🔄 Iniciando monitor de precios (WS klines)...")
        while self.running:
            try:
                if not self.monitored_symbols:
                    time.sleep(5); continue

                symbols = list(self.monitored_symbols)
                for symbol in symbols:
                    df1 = self.kws.get_dataframe(symbol, "1m", only_closed=False)
                    df5 = self.kws.get_dataframe(symbol, "5m", only_closed=False)
                    if not df1.empty and not df5.empty:
                        # DataCache espera columnas [timestamp, open, high, low, close, volume]
                        df1u = df1[['timestamp','open','high','low','close','volume']].copy()
                        df5u = df5[['timestamp','open','high','low','close','volume']].copy()
                        self.data_cache.update_data(symbol, df1u, df5u)
                time.sleep(0.8)  # pequeño respiro; ya no hacemos REST
            except Exception as e:
                logger.error(f"Error en monitor de precios (WS): {e}")
                time.sleep(2)


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

    def _rest_request(self, method, path, **kwargs):
        if time.time() < self.rest_circuit_until:
            return None  # circuito abierto por ban/limite

        url = f"{self.base_url}{path}"
        with self.rest_sema:
            try:
                resp = self.session.request(method, url, timeout=kwargs.get("timeout", 8), **kwargs)
            except Exception:
                return None

        # Manejo básico de rate-limit/ban
        if resp.status_code == 429 or "-1003" in (resp.text or ""):
            # Backoff: 60–120s según mensaje
            self.rest_circuit_until = time.time() + 90
            logger.error(f"⛔ Circuito REST abierto 90s por rate-limit/ban: {resp.text[:160]}")
            return None
        return resp

    def _max_leverage_for_symbol(self, symbol: str) -> Optional[int]:
        # lee una vez por arranque
        if hasattr(self, "_lev_brackets_cached") and self._lev_brackets_cached:
            pass
        else:
            resp = self._rest_request("GET", "/fapi/v1/leverageBracket")
            if not resp:
                return None
            self._lev_brackets = {b["symbol"]: b["brackets"] for b in resp.json()}
            self._lev_brackets_cached = True

        br = self._lev_brackets.get(symbol)
        if not br:
            return None
        # bracket 1 suele traer maxLeverage
        try:
            return int(br[0].get("initialLeverage", 20))
        except Exception:
            return None



    def open_trade(self, signal: Dict):
        """Abre nueva operación. En modo simulación no se envían órdenes a Binance"""
        try:
            symbol = signal['symbol']
            now = datetime.now()
            """"""
            last_opens = self.trade_open_history[symbol]
            last_opens = [ts for ts in last_opens if (now - ts).total_seconds() < 120]
            if len(last_opens) >= 200:
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

            # === NUEVO: usa TP/SL dinámicos si vienen en la señal ===
            tp = signal.get('tp')
            sl = signal.get('sl')
            tp_pct = signal.get('tp_pct')
            sl_pct = signal.get('sl_pct')

            # Tamaño por riesgo (si tenemos sl_pct dinámico)
            quantity = self.calculate_position_size(entry_price, sl_pct if sl_pct is not None else None)
            if quantity <= 0:
                logger.warning(f"⚠️ Cantidad inválida para {symbol}, saltando operación")
                return

            # Si por alguna razón no hubo TP/SL dinámicos, usa fijos de backup
            if tp is None or sl is None:
                if trade_type == "LONG":
                    tp = entry_price * (1 + TAKE_PROFIT_PCT / 100)
                    sl = entry_price * (1 - STOP_LOSS_PCT / 100)
                    sl_pct = STOP_LOSS_PCT
                else:
                    tp = entry_price * (1 - TAKE_PROFIT_PCT / 100)
                    sl = entry_price * (1 + STOP_LOSS_PCT / 100)
                    sl_pct = STOP_LOSS_PCT



           


            if not self.simulate:
                # En modo real se envían las órdenes
                max_lev = self._max_leverage_for_symbol(symbol)
                if trade_type == "LONG":
                    result = self.api.open_long_position(symbol=symbol, quantity=quantity, leverage= max_lev)
                else:
                    result = self.api.open_short_position(symbol=symbol, quantity=quantity, leverage= max_lev)
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
            trade = Trade(
                symbol=symbol,
                trade_type=trade_type,
                entry_price=entry_price,
                entry_time=datetime.now(),
                tp=tp,
                sl=sl,
                quantity=quantity,
                volatility_score=signal['volatility_score'],
                confidence=signal['confidence']
            )
            self.active_trades[symbol] = trade
            # Registrar la señal en el log de señales para análisis posterior, con datos relevantes
            self.signals_log.append({
                'symbol': symbol,
                'original_signal': trade_type,
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
        """Espera a que la posición se abra y coloca órdenes de TP/SL. En simulación se omite.
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
            logger.error(f"Error en wait_and_set_tp_sl: {e}")"""

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
                        # === NUEVO: refuerzo Markov con el resultado del trade ===
            self._update_markov_from_trade(symbol, trade.entry_price, exit_price)

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
                        # Antes de lanzar hilos de estrategia:
            self.kws = KlineWebSocketCache(
            pairs={sym: ["1m", "5m"] for sym in list(self.monitored_symbols)[:40]},
            max_candles=1500,
            include_open_candle=True,
            backfill_on_start=True,
            rest_limits={'1m': 1500, '5m': 1500},   # <- como en tu get_klines
        )
      

            self.kws.start()

                        # Cargar estado previo y arrancar autosave
            self.load_state()
            self.start_autosave()


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

    def export_learning_snapshot(self, folder: str = "snapshots"):
        try:
            Path(folder).mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Persistencia completa
            tmp_path = Path(folder) / f"state_{ts}.json"
            PersistenceManager(tmp_path).save(self)
            # Señales y trades a CSV
            if self.signals_log:
                pd.DataFrame(self.signals_log).to_csv(Path(folder) / f"signals_{ts}.csv", index=False)
            if self.completed_trades:
                pd.DataFrame(self.completed_trades).to_csv(Path(folder) / f"trades_{ts}.csv", index=False)
            logger.info(f"📦 Snapshot exportado en {folder}")
        except Exception as e:
            logger.error(f"No se pudo exportar snapshot: {e}")


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

                # Persistir y terminar autosave
        try:
            self.stop_autosave()
            self.save_state()
        except Exception as e:
            logger.error(f"Error guardando estado en cleanup: {e}")



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
