# binance_fixed.py - Versión corregida para errores de position side
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
import pandas as pd
import time
import logging
import math

from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, List, Union

class BinanceAPI:
        def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
            """
            Inicializa la conexión con Binance Futures API
            """
            if not api_key or not api_secret:
                raise ValueError("API Key y API Secret son requeridos")
            
            print(f"Inicializando API con Key: {api_key[:8]}...")
            self.client = Client(api_key, api_secret, testnet=testnet)
            self.testnet = testnet
            
            # Configurar logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            
            # Cache para información de símbolos
            self._symbol_info_cache = {}
            
            # Detectar automáticamente el modo de posición del usuario
            self._position_mode = None
            self._detect_position_mode()

            # Diccionario para almacenar IDs de órdenes TP y SL por símbolo
            # La estructura es: {symbol: {'tp': order_id_tp, 'sl': order_id_sl}}
            self.tp_sl_orders: Dict[str, Dict[str, Optional[int]]] = {}
            
        def _detect_position_mode(self):
            """Detecta automáticamente si el usuario tiene Hedge Mode activado"""
            try:
                # Obtener configuración actual de posición
                response = self.client.futures_get_position_mode()
                self._position_mode = "hedge" if response["dualSidePosition"] else "one_way"
                self.logger.info(f"Modo de posición detectado: {self._position_mode}")
            except Exception as e:
                self.logger.warning(f"No se pudo detectar el modo de posición: {e}")
                self._position_mode = "one_way"  # Default más común
        
        def set_position_mode(self, dual_side_position: bool) -> Optional[Dict]:
            """
            Cambia el modo de posición (Hedge Mode on/off)
            
            Args:
                dual_side_position: True para Hedge Mode, False para One-way Mode
            """
            try:
                result = self.client.futures_change_position_mode(
                    dualSidePosition=dual_side_position,
                    timestamp=int(time.time() * 1000)
                )
                mode = "Hedge Mode" if dual_side_position else "One-way Mode"
                self.logger.info(f"Modo de posición cambiado a: {mode}")
                self._position_mode = "hedge" if dual_side_position else "one_way"
                return result
            except Exception as e:
                error_msg = str(e)
                if "No need to change position side" in error_msg:
                    mode = "Hedge Mode" if dual_side_position else "One-way Mode"
                    self.logger.info(f"El modo de posición ya está en: {mode}")
                    self._position_mode = "hedge" if dual_side_position else "one_way"
                    return {"msg": f"Already in {mode}"}
                else:
                    self.logger.error(f"Error cambiando modo de posición: {e}")
                    return None
        
        def get_position_mode(self) -> str:
            """Retorna el modo de posición actual"""
            return self._position_mode
        
        def _get_correct_position_side(self, side: str, force_position_side: Optional[str] = None) -> str:
            """
            Determina el positionSide correcto según el modo de posición.
            
            Args:
                side: 'BUY' o 'SELL'
                force_position_side: 'BOTH', 'LONG' o 'SHORT' para forzar un side específico
            """
            # En One-way Mode siempre usar BOTH, ignorando cualquier force_position_side
            if self._position_mode != "hedge":
                return "BOTH"

            # En Hedge Mode, respetar force_position_side si se pasa
            if force_position_side:
                return force_position_side

            # En Hedge Mode sin force, usar LONG/SHORT según side
            return "LONG" if side == "BUY" else "SHORT"
    
        
        def _get_symbol_info(self, symbol: str) -> Dict:
            """Obtiene información del símbolo con cache"""
            if symbol not in self._symbol_info_cache:
                try:
                    info = self.client.futures_exchange_info()
                    for s in info['symbols']:
                        if s['symbol'] == symbol:
                            self._symbol_info_cache[symbol] = s
                            break
                except Exception as e:
                    self.logger.error(f"Error getting symbol info: {e}")
                    return {}
            return self._symbol_info_cache.get(symbol, {})
        
        def _round_quantity(self, symbol: str, quantity: float) -> str:
            """Redondea la cantidad según las reglas del símbolo"""
            symbol_info = self._get_symbol_info(symbol)
            if not symbol_info:
                return str(quantity)
            
            for filter in symbol_info.get('filters', []):
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    precision = len(str(step_size).split('.')[-1].rstrip('0'))
                    rounded = float(Decimal(str(quantity)).quantize(
                        Decimal(str(step_size)), rounding=ROUND_DOWN
                    ))
                    return f"{rounded:.{precision}f}".rstrip('0').rstrip('.')
            return str(quantity)
        
        def _get_position_quantity(self, symbol: str, position_side: Optional[str]) -> float:
            """
            Devuelve la cantidad absoluta de la posición abierta en 'symbol' y 'position_side'.
            """
            positions = self.client.futures_position_information(symbol=symbol)
            for p in positions:
                if p['positionSide'] == position_side:
                    return abs(float(p['positionAmt']))
            return 0.0

    
        def set_take_profit(self, symbol: str, take_profit_price: float, position_side: Optional[str] = None) -> Optional[Dict]:
            """
            Coloca una orden de take profit utilizando una orden condicional LIMIT (tipo TAKE_PROFIT).

            Se calcula la dirección (long/short) a partir de ``position_side``. En modo one-way,
            ``positionSide`` siempre será "BOTH", por lo que se determina la dirección
            interpretando ``position_side``. Para LONG se vende (side="SELL"); para SHORT se
            compra (side="BUY"). La orden se configura con stopPrice y price iguales por
            defecto para intentar salir exactamente al nivel objetivo. Se agrega lógica de
            reintento y auto-corrección para manejar errores comunes de la API (tick size,
            notional mínimo, triggers inmediatos, etc.). Además, se almacena el ID de la orden
            en ``self.tp_sl_orders[symbol]['tp']`` para su seguimiento posterior.
            """
            try:
                # Obtener precio de mercado actual y filtros
                ticker = self.get_ticker_price(symbol)
                if not ticker:
                    self.logger.error(f"No se pudo obtener precio actual de {symbol} para TP")
                    return None
                mark_price = float(ticker['price'])
                symbol_info = self._get_symbol_info(symbol)
                tick_size = next(
                    float(f['tickSize']) for f in symbol_info.get('filters', [])
                    if f.get('filterType') == 'PRICE_FILTER'
                )

                # Buffer para evitar triggers inmediatos
                buffer = max(mark_price * 0.00001, tick_size * 10)
                direction = (position_side or "LONG").upper()
                # Determinar objetivo de precio y side
                if direction == "LONG":
                    # En LONG vendemos, stopPrice debe estar por encima del precio actual
                    target_price = max(take_profit_price, mark_price + buffer)
                    side = "SELL"
                elif direction == "SHORT":
                    # En SHORT compramos, stopPrice debe estar por debajo del precio actual
                    target_price = min(take_profit_price, mark_price - buffer)
                    side = "BUY"
                else:
                    target_price = max(take_profit_price, mark_price + buffer)
                    side = "SELL"

                # Redondear precios
                stop_price = float(self._round_price_limit(symbol, target_price))
                # Para TP usamos el mismo precio límite por defecto. Para ventas en long es
                # recomendable un poco por debajo del trigger, pero aquí se mantiene igual para
                # precisión y se ajustará en reintentos si es necesario.
                price = stop_price

                # Obtener cantidad de la posición existente (BOTH en modo one-way)
                qty = self._get_position_quantity(symbol, "BOTH")
                if qty <= 0:
                    self.logger.warning(f"No hay posición abierta para colocar TP en {symbol}")
                    return None

                # Preparar parámetros de orden
                order_params = {
                    'symbol': symbol,
                    'side': side,
                    'type': 'TAKE_PROFIT',
                    'stopPrice': str(stop_price),
                    'price': str(price),
                    'quantity': qty,
                    'timeInForce': 'GTC',
                    'reduceOnly': True,
                    'positionSide': 'BOTH',
                    # newClientOrderId se añadirá en _safe_futures_create_order
                }

                order = self._safe_futures_create_order(order_params)
                if order:
                    order_id = order.get('orderId')
                    # Registrar ID de TP
                    if symbol not in self.tp_sl_orders:
                        self.tp_sl_orders[symbol] = {'tp': None, 'sl': None}
                    self.tp_sl_orders[symbol]['tp'] = order_id
                    self.logger.info(
                        f"TP LIMIT colocado en {stop_price} para {symbol} (side: {side}, orderId: {order_id})"
                    )
                    return order
                else:
                    return None

            except Exception as e:
                self.logger.error(f"Error placing take profit: {e}")
                return None

        def set_stop_loss(self, symbol: str, stop_price: float, position_side: Optional[str] = None) -> Optional[Dict]:
            """
            Coloca una orden de stop loss utilizando una orden condicional LIMIT (tipo STOP).

            Para posiciones LONG se vende (side="SELL") cuando el precio cae por debajo de
            ``stop_price``. En este caso el precio límite se sitúa ligeramente por debajo de
            ``stop_price`` para aumentar la probabilidad de llenado. Para posiciones SHORT se
            compra (side="BUY") cuando el precio sube por encima de ``stop_price``, y el precio
            límite se fija un poco por encima del disparador. La función incluye manejo de
            errores con reintentos, ajustes automáticos y almacenamiento de IDs en
            ``self.tp_sl_orders[symbol]['sl']``.
            """
            try:
                ticker = self.get_ticker_price(symbol)
                if not ticker:
                    self.logger.error(f"No se pudo obtener precio actual de {symbol} para SL")
                    return None
                mark_price = float(ticker['price'])
                symbol_info = self._get_symbol_info(symbol)
                tick_size = next(
                    float(f['tickSize']) for f in symbol_info.get('filters', [])
                    if f.get('filterType') == 'PRICE_FILTER'
                )
                buffer = max(mark_price * 0.00001, tick_size * 10)
                direction = (position_side or "LONG").upper()
                # Definir target_price y side para trigger
                if direction == "LONG":
                    # En LONG vendemos; stopPrice debe estar por debajo del precio de mercado
                    target_price = min(stop_price, mark_price - buffer)
                    side = "SELL"
                elif direction == "SHORT":
                    # En SHORT compramos; stopPrice debe estar por encima del precio de mercado
                    target_price = max(stop_price, mark_price + buffer)
                    side = "BUY"
                else:
                    target_price = min(stop_price, mark_price - buffer)
                    side = "SELL"

                # Cálculo de stopPrice redondeado
                stop_price_rounded = float(self._round_price_limit(symbol, target_price))
                # Calcular precio límite: ligeramente peor que stopPrice para asegurar fill
                if direction == "LONG":
                    # SL en long: vendemos un poco más abajo que stopPrice
                    limit_price = stop_price_rounded * 0.9995
                else:
                    # SL en short: compramos un poco más arriba que stopPrice
                    limit_price = stop_price_rounded * 1.0005
                price_rounded = float(self._round_price_limit(symbol, limit_price))

                qty = self._get_position_quantity(symbol, "BOTH")
                if qty <= 0:
                    self.logger.warning(f"No hay posición abierta para colocar SL en {symbol}")
                    return None

                order_params = {
                    'symbol': symbol,
                    'side': side,
                    'type': 'STOP',
                    'stopPrice': str(stop_price_rounded),
                    'price': str(price_rounded),
                    'quantity': qty,
                    'timeInForce': 'GTC',
                    'reduceOnly': True,
                    'positionSide': 'BOTH',
                    # newClientOrderId se asignará en _safe_futures_create_order
                }

                order = self._safe_futures_create_order(order_params)
                if order:
                    order_id = order.get('orderId')
                    # Registrar ID de SL
                    if symbol not in self.tp_sl_orders:
                        self.tp_sl_orders[symbol] = {'tp': None, 'sl': None}
                    self.tp_sl_orders[symbol]['sl'] = order_id
                    self.logger.info(
                        f"SL LIMIT colocado en {stop_price_rounded} para {symbol} (side: {side}, orderId: {order_id})"
                    )
                    return order
                else:
                    return None

            except Exception as e:
                self.logger.error(f"Error placing stop loss: {e}")
                return None
            

        def _generate_order_id(self, order_type, symbol):
            base = f"{order_type[:2]}_{symbol[:10]}_{int(time.time()*1000)%10**8}"
            return base[:36]




        def _safe_futures_create_order(self, order_params: Dict, max_retries: int = 3) -> Optional[Dict]:
            """
            Crea órdenes de futuros de forma robusta realizando reintentos y correcciones
            automáticas basadas en el código de error devuelto por Binance. Utiliza
            ``newClientOrderId`` para mantener idempotencia entre reintentos.

            Args:
                order_params: diccionario de parámetros para futures_create_order
                max_retries: número máximo de intentos

            Returns:
                La respuesta de la API en caso de éxito o ``None`` si falla después de los
                reintentos.
            """
            # Generar un identificador único para el cliente si no se proporcionó.
            base_id = order_params.get('newClientOrderId')
            if not base_id:
                base_id = self._generate_order_id(order_params['type'], order_params['symbol'])
                order_params['newClientOrderId'] = base_id
            elif len(base_id) > 36:
                base_id = base_id[:36]
                order_params['newClientOrderId'] = base_id

            attempt = 1
            while attempt <= max_retries:
                try:
                    return self.client.futures_create_order(**order_params)
                except BinanceAPIException as e:
                    code = e.code
                    msg = str(e)
                    self.logger.warning(f"Intento {attempt}: error al crear orden ({code}) {msg}")
                    symbol = order_params.get('symbol')
                    # Ajustar tick size y precios

                    if code == -4014 or "Price not increased by tick size" in msg:
                        symbol = order_params.get('symbol')
                        symbol_info = self._get_symbol_info(symbol)
                        tick_size = next(
                            float(f['tickSize']) for f in symbol_info.get('filters', [])
                            if f.get('filterType') == 'PRICE_FILTER'
                        )
                        for price_field in ['price', 'stopPrice']:
                            if price_field in order_params:
                                price = float(order_params[price_field])
                                # Redondea al tick más cercano hacia abajo
                                price = math.floor(price / tick_size) * tick_size
                                # Incrementa/decrementa 1 tick
                                if order_params['side'] == 'BUY':
                                    price += tick_size
                                elif order_params['side'] == 'SELL':
                                    price -= tick_size
                                price = round(price, int(abs(math.log10(tick_size))))
                                order_params[price_field] = price



                    if code in (-4015, -4022) or 'Tick size' in msg or 'PRICE_FILTER' in msg:
                        if 'price' in order_params:
                            try:
                                order_params['price'] = self._round_price_limit(symbol, float(order_params['price']))
                            except Exception:
                                pass
                        if 'stopPrice' in order_params:
                            try:
                                order_params['stopPrice'] = self._round_price_limit(symbol, float(order_params['stopPrice']))
                            except Exception:
                                pass
                    # Ajustar notional mínimo
                    if code == -4164 or 'notional' in msg:
                        order_params['reduceOnly'] = True
                    # Trigger immediato: ajustamos stopPrice alejándolo del precio de mercado
                    if code == -2021 or 'immediately trigger' in msg:
                        ticker = self.get_ticker_price(symbol)
                        if ticker:
                            market_price = float(ticker['price'])
                            if order_params['type'] in ['TAKE_PROFIT', 'TAKE_PROFIT_MARKET']:
                                if order_params['side'] == 'SELL':
                                    new_sp = max(float(order_params['stopPrice']), market_price * 1.0005)
                                    order_params['stopPrice'] = self._round_price_limit(symbol, new_sp)
                                    order_params['price'] = order_params['stopPrice']
                                else:
                                    new_sp = min(float(order_params['stopPrice']), market_price * 0.9995)
                                    order_params['stopPrice'] = self._round_price_limit(symbol, new_sp)
                                    order_params['price'] = order_params['stopPrice']
                            elif order_params['type'] in ['STOP', 'STOP_MARKET']:
                                if order_params['side'] == 'SELL':
                                    new_sp = min(float(order_params['stopPrice']), market_price * 0.9995)
                                    order_params['stopPrice'] = self._round_price_limit(symbol, new_sp)
                                    limit_price = new_sp * 0.9995
                                    order_params['price'] = self._round_price_limit(symbol, limit_price)
                                else:
                                    new_sp = max(float(order_params['stopPrice']), market_price * 1.0005)
                                    order_params['stopPrice'] = self._round_price_limit(symbol, new_sp)
                                    limit_price = new_sp * 1.0005
                                    order_params['price'] = self._round_price_limit(symbol, limit_price)
                    # ReduceOnly rechazado: probablemente no hay posición abierta
                    if code == -2022 or 'reduceOnly' in msg:
                        self.logger.warning(f"Orden reduceOnly rechazada, probablemente no hay posición activa para {symbol}")
                        return None
                    attempt += 1
                    time.sleep(0.5 * attempt)
                except Exception as e2:
                    self.logger.error(f"Error inesperado al crear orden: {e2}")
                    return None
            self.logger.error(f"No se pudo colocar la orden después de {max_retries} intentos")
            return None
        # ======================== FORMATEO DE PRECIOS ========================
        def _round_price(self, symbol: str, price: float) -> str:
            """
            Redondea el precio usando la misma precisión decimal que el precio de mercado actual
            """
            try:
                # Obtener precio de mercado actual
                ticker = self.get_ticker_price(symbol)
                market_price_str = str(float(ticker['price']))
                
                # Determinar número de decimales del precio de mercado
                if '.' in market_price_str:
                    decimal_places = len(market_price_str.split('.')[1].rstrip('0'))
                    # Si no hay decimales significativos, usar al menos 2
                    if decimal_places == 0:
                        decimal_places = 2
                else:
                    decimal_places = 2
                
                # Aplicar la misma precisión al precio dado
                formatted_price = f"{price:.{decimal_places}f}"
                
                self.logger.debug(f"Price formatting for {symbol}: {price} -> {formatted_price} (market precision: {decimal_places})")
                return formatted_price
                
            except Exception as e:
                self.logger.error(f"Error formatting price for {symbol}: {e}")
                # Fallback: usar 6 decimales por defecto
                return f"{price:.6f}".rstrip('0').rstrip('.')
            
        def _round_price_limit(self, symbol: str, price: float) -> str:
            """
            Redondea `price` siguiendo la precisión que indica PRICE_FILTER.tickSize.
            """
            try:
                # 1) Obtén el tickSize
                symbol_info = self._get_symbol_info(symbol)
                tick_size = next(
                    Decimal(f['tickSize'])
                    for f in symbol_info.get('filters', [])
                    if f.get('filterType') == 'PRICE_FILTER'
                )
                
                # 2) Calcula el exponente (–log10(tick_size)) para saber decimales
                #    Ej: tick_size=0.001 -> precision=3
                precision = int(-tick_size.as_tuple().exponent)
                
                # 3) Cuantiza el precio al múltiplo de tick_size (hacia abajo)
                rounded = (Decimal(price)
                        .quantize(tick_size, rounding=ROUND_DOWN))
                
                # 4) Formatea a string con la precisión exacta
                formatted = format(rounded, f'.{precision}f')
                
                self.logger.debug(
                    f"_round_price {symbol}: {price} → {formatted} "
                    f"(tickSize={tick_size}, precision={precision})"
                )
                return formatted

            except Exception as e:
                self.logger.error(f"Error formatting price for {symbol}: {e}")
                # Fallback seguro a 8 decimales
                return f"{price:.8f}"

    # ======================== CONFIGURACIÓN DE CUENTA ========================
        
        def set_leverage(self, symbol: str, leverage: int) -> Optional[Dict]:
            """Establece el apalancamiento para un símbolo"""
            try:
                result = self.client.futures_change_leverage(
                    symbol=symbol,
                    leverage=leverage,
                    timestamp=int(time.time() * 999)
                )
                self.logger.info(f"Leverage set to {leverage}x for {symbol}")
                return result
            except Exception as e:
                self.logger.error(f"Error setting leverage: {e}")
                return None
        
        def set_margin_type(self, symbol: str, margin_type: str) -> Optional[Dict]:
            """Establece el tipo de margen (ISOLATED o CROSSED)"""
            try:
                result = self.client.futures_change_margin_type(
                    symbol=symbol,
                    marginType=margin_type,
                    timestamp=int(time.time() * 999)
                )
                self.logger.info(f"Margin type set to {margin_type} for {symbol}")
                return result
            except Exception as e:
                error_msg = str(e)
                if "No need to change margin type" in error_msg:
                    self.logger.info(f"El tipo de margen ya está en {margin_type} para {symbol}")
                    return {"msg": f"Already in {margin_type} mode"}
                else:
                    self.logger.error(f"Error setting margin type: {e}")
                    return None

        # ======================== INFORMACIÓN DE MERCADO ========================
        
        def get_ticker_price(self, symbol: str) -> Optional[Dict]:
            """Obtiene el precio actual del símbolo"""
            try:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                return ticker
            except Exception as e:
                self.logger.error(f"Error getting ticker price: {e}")
                return None
        
        def get_position_info(self, symbol: Optional[str] = None) -> Optional[Union[Dict, List]]:
            """Obtiene información de posiciones"""
            try:
                if symbol:
                    positions = self.client.futures_position_information(symbol=symbol)
                    # Retornar todas las posiciones del símbolo (puede haber LONG y SHORT en hedge mode)
                    relevant_positions = []
                    for position in positions:
                        if position['symbol'] == symbol and float(position['positionAmt']) != 0:
                            relevant_positions.append(position)
                    return relevant_positions if relevant_positions else None
                else:
                    return self.client.futures_position_information()
            except Exception as e:
                self.logger.error(f"Error getting position info: {e}")
                return None
        
        def get_account_info(self) -> Optional[Dict]:
            """Obtiene información general de la cuenta"""
            try:
                return self.client.futures_account()
            except Exception as e:
                self.logger.error(f"Error getting account info: {e}")
                return None
        
        def get_open_orders(self, symbol: Optional[str] = None) -> Optional[List]:
            """Obtiene todas las órdenes abiertas"""
            try:
                if symbol:
                    orders = self.client.futures_get_open_orders()
                else:
                    orders = self.client.futures_get_open_orders()
                return orders
            except Exception as e:
                self.logger.error(f"Error getting open orders: {e}")
                return None

        # ======================== ÓRDENES ========================
        
        def create_market_order(self, symbol: str, side: str, quantity: float,
                                position_side: Optional[str] = None, reduce_only: bool = False) -> Optional[Dict]:
            """
            Crea una orden de mercado con position_side automático.
            
            Args:
                symbol: Símbolo del par
                side: 'BUY' o 'SELL'
                quantity: Cantidad (se asume ya valida)
                position_side: 'BOTH', 'LONG', 'SHORT' (automático si no se especifica)
                reduce_only: True para cerrar posición únicamente
            """
            try:
                # Obtener precio de mercado (por si necesitas usarlo en otra lógica)
                ticker = self.get_ticker_price(symbol)
                mark_price = float(ticker['price'])

                # Redondeo de cantidad y cálculo de positionSide
                quantity_str = self._round_quantity(symbol, quantity)
                correct_position_side = self._get_correct_position_side(side, position_side)

                # Parámetros de la orden
                order_params = {
                    'symbol': symbol,
                    'side': side,
                    'type': 'MARKET',
                    'quantity': quantity_str,
                    'positionSide': correct_position_side,
                    'timestamp': int(time.time() * 1000)
                }
                if reduce_only:
                    order_params['reduceOnly'] = 'true'

                # Creación de la orden
                order = self.client.futures_create_order(**order_params)
                self.logger.info(
                    f"Market order created: {side} {quantity_str} {symbol} "
                    f"(positionSide: {correct_position_side})"
                )
                return order

            except Exception as e:
                self.logger.error(f"Error creating market order: {e}")
                return None
    
        def create_limit_order(self, symbol: str, side: str, quantity: float, price: float,
                            position_side: Optional[str] = None, time_in_force: str = 'GTC',
                            reduce_only: bool = False) -> Optional[Dict]:
            """Crea una orden límite con position_side automático"""
            try:
                quantity_str = self._round_quantity(symbol, quantity)
                price_str = self._round_price(symbol, price)
                correct_position_side = self._get_correct_position_side(side, position_side)
                
                order_params = {
                    'symbol': symbol,
                    'side': side,
                    'type': 'LIMIT',
                    'quantity': quantity_str,
                    'price': price_str,
                    'timeInForce': time_in_force,
                    'positionSide': correct_position_side,
                    'timestamp': int(time.time() * 999)
                }
                
                if reduce_only:
                    order_params['reduceOnly'] = 'true'
                
                order = self.client.futures_create_order(**order_params)
                self.logger.info(f"Limit order created: {side} {quantity_str} {symbol} @ {price_str}")
                return order
            except Exception as e:
                self.logger.error(f"Error creating limit order: {e}")
                return None

        # ======================== FUNCIONES DE CONVENIENCIA ========================
        
        def open_long_position(self, symbol: str, quantity: float, leverage: Optional[int] = None) -> Optional[Dict]:
            """Abre una posición larga"""
            if leverage:
                self.set_leverage(symbol, leverage)
            
            return self.create_market_order(symbol, 'BUY', quantity)
        
        def open_short_position(self, symbol: str, quantity: float, leverage: Optional[int] = None) -> Optional[Dict]:
            """Abre una posición corta"""
            if leverage:
                self.set_leverage(symbol, leverage)
            
            return self.create_market_order(symbol, 'SELL', quantity)
        
        def close_all_positions(self, symbol: str) -> List[Optional[Dict]]:
            """Cierra todas las posiciones de un símbolo"""
            positions = self.get_position_info(symbol)
            results = []
            
            if not positions:
                self.logger.info(f"No hay posiciones abiertas para {symbol}")
                return results
            
            # positions es una lista en el caso corregido
            if isinstance(positions, dict):
                positions = [positions]
            
            for position in positions:
                position_amt = float(position['positionAmt'])
                if position_amt != 0:
                    quantity = abs(position_amt)
                    side = 'SELL' if position_amt > 0 else 'BUY'
                    position_side = position['positionSide']
                    
                    result = self.create_market_order(
                        symbol, side, quantity, position_side, reduce_only=True
                    )
                    results.append(result)
                    
                    if result:
                        self.logger.info(f"Closed position: {position_side} {quantity} {symbol}")
            
            return results
        
        def get_position_summary(self, symbol: str) -> Dict:
            """Obtiene un resumen de las posiciones"""
            positions = self.get_position_info(symbol)
            summary = {
                'symbol': symbol,
                'mode': self._position_mode,
                'positions': [],
                'total_pnl': 0.0
            }
            
            if positions:
                if isinstance(positions, dict):
                    positions = [positions]
                
                for pos in positions:
                    if float(pos['positionAmt']) != 0:
                        pos_info = {
                            'side': pos['positionSide'],
                            'size': float(pos['positionAmt']),
                            'entry_price': float(pos['entryPrice']),
                            'mark_price': float(pos['markPrice']),
                            'pnl': float(pos['unRealizedProfit']),
                            'percentage': float(pos['percentage'])
                        }
                        summary['positions'].append(pos_info)
                        summary['total_pnl'] += pos_info['pnl']
            
            return summary
        
        
            # Funciones LIMIT para agregar a la clase BinanceAPI
        # Agregar estas funciones a tu clase BinanceAPI en binance_api_mejorado.py
        def limit_open_long(self, symbol: str, quantity: float, limit_price: float, 
                        leverage: Optional[int] = None, 
                        time_in_force: str = 'GTC') -> Optional[Dict]:
            """
            Abre posición LONG con orden LIMIT
            
            Args:
                symbol: Par de trading (ej: "BTCUSDT")
                quantity: Cantidad a comprar
                limit_price: Precio límite máximo para la compra
                leverage: Apalancamiento (opcional)
                time_in_force: Tipo de validez ('GTC', 'IOC', 'FOK')
            
            Returns:
                Dict con información de la orden o None si falla
            """
            try:
                # Configurar leverage si se especifica
                if leverage:
                    leverage_result = self.set_leverage(symbol, leverage)
                    if not leverage_result:
                        self.logger.warning(f"No se pudo configurar leverage {leverage}x para {symbol}")
                
                # Obtener precio actual para validación
                ticker = self.get_ticker_price(symbol)
                if not ticker:
                    self.logger.error(f"No se pudo obtener precio actual de {symbol}")
                    return None
                
                current_price = float(ticker['price'])
                
                # Validar que el precio límite sea menor al actual (para LONG)
                if limit_price >= current_price:
                    self.logger.warning(
                        f"Precio límite ${limit_price:.4f} debe ser menor al actual ${current_price:.4f} para LONG"
                    )
                    # Auto-ajustar a 0.1% por debajo del precio actual
                    limit_price = current_price * 0.99999
                    self.logger.info(f"Auto-ajustando precio límite a ${limit_price:.4f}")
                
                # Redondear cantidad y precio
                quantity_str = self._round_quantity(symbol, quantity)
                price_str = self._round_price(symbol, limit_price)
                
                # Obtener positionSide correcto
                correct_position_side = self._get_correct_position_side('BUY')
                
                # Crear orden LIMIT de compra
                order_params = {
                    'symbol': symbol,
                    'side': 'BUY',
                    'type': 'LIMIT',
                    'quantity': quantity_str,
                    'price': price_str,
                    'timeInForce': time_in_force,
                    'positionSide': correct_position_side,
                    'timestamp': int(time.time() * 1000)
                }
                
                order = self.client.futures_create_order(**order_params)
                
                if order:
                    order_id = order['orderId']
                    self.logger.info(f"LONG LIMIT creada: {symbol} | {quantity_str} @ ${price_str}")
                    return order
                
            except Exception as e:
                self.logger.error(f"Error abriendo LONG LIMIT para {symbol}: {e}")
                return None

        def limit_open_short(self, symbol: str, quantity: float, limit_price: float, 
                            leverage: Optional[int] = None, 
                            time_in_force: str = 'GTC') -> Optional[Dict]:
            """
            Abre posición SHORT con orden LIMIT
            
            Args:
                symbol: Par de trading (ej: "BTCUSDT")
                quantity: Cantidad a vender
                limit_price: Precio límite mínimo para la venta
                leverage: Apalancamiento (opcional)
                time_in_force: Tipo de validez ('GTC', 'IOC', 'FOK')
            
            Returns:
                Dict con información de la orden o None si falla
            """
            try:
                # Configurar leverage si se especifica
                if leverage:
                    leverage_result = self.set_leverage(symbol, leverage)
                    if not leverage_result:
                        self.logger.warning(f"No se pudo configurar leverage {leverage}x para {symbol}")
                
                # Obtener precio actual para validación
                ticker = self.get_ticker_price(symbol)
                if not ticker:
                    self.logger.error(f"No se pudo obtener precio actual de {symbol}")
                    return None
                
                current_price = float(ticker['price'])
                
                # Validar que el precio límite sea mayor al actual (para SHORT)
                if limit_price <= current_price:
                    self.logger.warning(
                        f"Precio límite ${limit_price:.4f} debe ser mayor al actual ${current_price:.4f} para SHORT"
                    )
                    # Auto-ajustar a 0.1% por encima del precio actual
                    limit_price = current_price * 1.00001
                    self.logger.info(f"Auto-ajustando precio límite a ${limit_price:.4f}")
                
                # Redondear cantidad y precio
                quantity_str = self._round_quantity(symbol, quantity)
                price_str = self._round_price(symbol, limit_price)
                
                # Obtener positionSide correcto
                correct_position_side = self._get_correct_position_side('SELL')
                
                # Crear orden LIMIT de venta
                order_params = {
                    'symbol': symbol,
                    'side': 'SELL',
                    'type': 'LIMIT',
                    'quantity': quantity_str,
                    'price': price_str,
                    'timeInForce': time_in_force,
                    'positionSide': correct_position_side,
                    'timestamp': int(time.time() * 1000)
                }
                
                order = self.client.futures_create_order(**order_params)
                
                if order:
                    order_id = order['orderId']
                    self.logger.info(f"SHORT LIMIT creada: {symbol} | {quantity_str} @ ${price_str}")
                    return order
                
            except Exception as e:
                self.logger.error(f"Error abriendo SHORT LIMIT para {symbol}: {e}")
                return None

        def limit_exit_long(self, symbol: str, quantity: Optional[float] = None, 
                        limit_price: Optional[float] = None, 
                        time_in_force: str = 'GTC') -> Optional[Dict]:
            """
            Cierra posición LONG con orden LIMIT (vende)
            
            Args:
                symbol: Par de trading
                quantity: Cantidad a cerrar (None para cerrar toda la posición)
                limit_price: Precio límite mínimo de venta (None para precio actual + 0.1%)
                time_in_force: Tipo de validez
            
            Returns:
                Dict con información de la orden o None si falla
            """
            try:
                # Obtener información de la posición si no se especifica cantidad
                if quantity is None:
                    position_info = self.get_position_info(symbol)
                    if not position_info:
                        self.logger.error(f"No se encontró posición LONG para {symbol}")
                        return None
                    
                    # Manejar si position_info es una lista o dict
                    if isinstance(position_info, list):
                        long_position = None
                        for pos in position_info:
                            if float(pos['positionAmt']) > 0:  # Posición LONG
                                long_position = pos
                                break
                        if not long_position:
                            self.logger.error(f"No se encontró posición LONG activa para {symbol}")
                            return None
                        position_amt = float(long_position['positionAmt'])
                    else:
                        position_amt = float(position_info['positionAmt'])
                        if position_amt <= 0:
                            self.logger.error(f"No hay posición LONG para cerrar en {symbol}")
                            return None
                    
                    quantity = abs(position_amt)
                
                # Obtener precio actual si no se especifica límite
                if limit_price is None:
                    ticker = self.get_ticker_price(symbol)
                    if not ticker:
                        self.logger.error(f"No se pudo obtener precio actual de {symbol}")
                        return None
                    current_price = float(ticker['price'])
                    # Para salir de LONG, queremos vender a precio igual o mayor
                    limit_price = current_price * 1.001  # 0.1% por encima
                
                # Validar precio límite
                ticker = self.get_ticker_price(symbol)
                if ticker:
                    current_price = float(ticker['price'])
                    if limit_price < current_price * 0.95:  # Más de 5% por debajo del actual
                        self.logger.warning(
                            f"Precio límite ${limit_price:.4f} muy bajo comparado con actual ${current_price:.4f}"
                        )
                
                # Redondear cantidad y precio
                quantity_str = self._round_quantity(symbol, quantity)
                price_str = self._round_price(symbol, limit_price)
                
                # Obtener positionSide correcto para cerrar LONG
                correct_position_side = self._get_correct_position_side('SELL', 'LONG')
                
                # Crear orden LIMIT de venta para cerrar LONG
                order_params = {
                    'symbol': symbol,
                    'side': 'SELL',
                    'type': 'LIMIT',
                    'quantity': quantity_str,
                    'price': price_str,
                    'timeInForce': time_in_force,
                    'reduceOnly': 'true',
                    'positionSide': correct_position_side,
                    'timestamp': int(time.time() * 1000)
                }
                
                order = self.client.futures_create_order(**order_params)
                
                if order:
                    order_id = order['orderId']
                    self.logger.info(f"EXIT LONG LIMIT creada: {symbol} | {quantity_str} @ ${price_str}")
                    return order
                
            except Exception as e:
                self.logger.error(f"Error cerrando LONG LIMIT para {symbol}: {e}")
                return None

        def limit_exit_short(self, symbol: str, quantity: Optional[float] = None, 
                            limit_price: Optional[float] = None, 
                            time_in_force: str = 'GTC') -> Optional[Dict]:
            """
            Cierra posición SHORT con orden LIMIT (compra)
            
            Args:
                symbol: Par de trading
                quantity: Cantidad a cerrar (None para cerrar toda la posición)
                limit_price: Precio límite máximo de compra (None para precio actual - 0.1%)
                time_in_force: Tipo de validez
            
            Returns:
                Dict con información de la orden o None si falla
            """
            try:
                # Obtener información de la posición si no se especifica cantidad
                if quantity is None:
                    position_info = self.get_position_info(symbol)
                    if not position_info:
                        self.logger.error(f"No se encontró posición SHORT para {symbol}")
                        return None
                    
                    # Manejar si position_info es una lista o dict
                    if isinstance(position_info, list):
                        short_position = None
                        for pos in position_info:
                            if float(pos['positionAmt']) < 0:  # Posición SHORT
                                short_position = pos
                                break
                        if not short_position:
                            self.logger.error(f"No se encontró posición SHORT activa para {symbol}")
                            return None
                        position_amt = float(short_position['positionAmt'])
                    else:
                        position_amt = float(position_info['positionAmt'])
                        if position_amt >= 0:
                            self.logger.error(f"No hay posición SHORT para cerrar en {symbol}")
                            return None
                    
                    quantity = abs(position_amt)
                
                # Obtener precio actual si no se especifica límite
                if limit_price is None:
                    ticker = self.get_ticker_price(symbol)
                    if not ticker:
                        self.logger.error(f"No se pudo obtener precio actual de {symbol}")
                        return None
                    current_price = float(ticker['price'])
                    # Para salir de SHORT, queremos comprar a precio igual o menor
                    limit_price = current_price * 0.999  # 0.1% por debajo
                
                # Validar precio límite
                ticker = self.get_ticker_price(symbol)
                if ticker:
                    current_price = float(ticker['price'])
                    if limit_price > current_price * 1.05:  # Más de 5% por encima del actual
                        self.logger.warning(
                            f"Precio límite ${limit_price:.4f} muy alto comparado con actual ${current_price:.4f}"
                        )
                
                # Redondear cantidad y precio
                quantity_str = self._round_quantity(symbol, quantity)
                price_str = self._round_price(symbol, limit_price)
                
                # Obtener positionSide correcto para cerrar SHORT
                correct_position_side = self._get_correct_position_side('BUY', 'SHORT')
                
                # Crear orden LIMIT de compra para cerrar SHORT
                order_params = {
                    'symbol': symbol,
                    'side': 'BUY',
                    'type': 'LIMIT',
                    'quantity': quantity_str,
                    'price': price_str,
                    'timeInForce': time_in_force,
                    'reduceOnly': 'true',
                    'positionSide': correct_position_side,
                    'timestamp': int(time.time() * 1000)
                }
                
                order = self.client.futures_create_order(**order_params)
                
                if order:
                    order_id = order['orderId']
                    self.logger.info(f"EXIT SHORT LIMIT creada: {symbol} | {quantity_str} @ ${price_str}")
                    return order
                
            except Exception as e:
                self.logger.error(f"Error cerrando SHORT LIMIT para {symbol}: {e}")
                return None

        def cancel_limit_long(self, symbol: str, order_id: Optional[str] = None) -> Optional[Dict]:
            """
            Cancela orden LIMIT de apertura LONG específica o todas las órdenes LONG del símbolo
            
            Args:
                symbol: Par de trading
                order_id: ID específico de la orden (None para cancelar todas las LONG)
            
            Returns:
                Dict con información de cancelación o None si falla
            """
            try:
                # Si se especifica order_id, cancelar solo esa orden
                if order_id:
                    cancel_result = self.client.futures_cancel_order(
                        symbol=symbol,
                        orderId=order_id,
                        timestamp=int(time.time() * 1000)
                    )
                    self.logger.info(f"Orden LONG cancelada: {symbol} | ID: {order_id}")
                    return cancel_result
                
                # Si no se especifica order_id, cancelar todas las órdenes LONG del símbolo
                open_orders = self.get_open_orders(symbol)
                if not open_orders:
                    self.logger.info(f"No hay órdenes abiertas para {symbol}")
                    return None
                
                cancelled_orders = []
                long_orders_found = 0
                
                for order in open_orders:
                    # Identificar órdenes LONG (BUY para abrir, SELL para cerrar con reduceOnly)
                    is_long_open = (order['side'] == 'BUY' and 
                                order.get('reduceOnly', False) == False and
                                order['type'] in ['LIMIT'])
                    
                    is_long_close = (order['side'] == 'SELL' and 
                                    order.get('reduceOnly', False) == True and
                                    order['type'] in ['LIMIT'])
                    
                    if is_long_open or is_long_close:
                        long_orders_found += 1
                        try:
                            cancel_result = self.client.futures_cancel_order(
                                symbol=symbol,
                                orderId=order['orderId'],
                                timestamp=int(time.time() * 1000)
                            )
                            cancelled_orders.append({
                                'orderId': order['orderId'],
                                'side': order['side'],
                                'type': order['type'],
                                'price': order['price'],
                                'quantity': order['origQty'],
                                'result': 'CANCELLED'
                            })
                            self.logger.info(f"Orden LONG cancelada: {order['side']} {symbol} @ ${order['price']}")
                        except Exception as e:
                            cancelled_orders.append({
                                'orderId': order['orderId'],
                                'result': 'ERROR',
                                'error': str(e)
                            })
                
                if long_orders_found == 0:
                    self.logger.info(f"No se encontraron órdenes LONG para {symbol}")
                    return None
                
                summary = {
                    'symbol': symbol,
                    'total_long_orders_found': long_orders_found,
                    'cancelled_orders': cancelled_orders,
                    'successful_cancellations': len([o for o in cancelled_orders if o['result'] == 'CANCELLED'])
                }
                
                self.logger.info(f"Cancelación LONG completada: {symbol} | "
                                f"{summary['successful_cancellations']}/{long_orders_found} órdenes canceladas")
                
                return summary
                
            except Exception as e:
                self.logger.error(f"Error cancelando órdenes LONG para {symbol}: {e}")
                return None

        def cancel_limit_short(self, symbol: str, order_id: Optional[str] = None) -> Optional[Dict]:
            """
            Cancela orden LIMIT de apertura SHORT específica o todas las órdenes SHORT del símbolo
            
            Args:
                symbol: Par de trading
                order_id: ID específico de la orden (None para cancelar todas las SHORT)
            
            Returns:
                Dict con información de cancelación o None si falla
            """
            try:
                # Si se especifica order_id, cancelar solo esa orden
                if order_id:
                    cancel_result = self.client.futures_cancel_order(
                        symbol=symbol,
                        orderId=order_id,
                        timestamp=int(time.time() * 1000)
                    )
                    self.logger.info(f"Orden SHORT cancelada: {symbol} | ID: {order_id}")
                    return cancel_result
                
                # Si no se especifica order_id, cancelar todas las órdenes SHORT del símbolo
                open_orders = self.get_open_orders(symbol)
                if not open_orders:
                    self.logger.info(f"No hay órdenes abiertas para {symbol}")
                    return None
                
                cancelled_orders = []
                short_orders_found = 0
                
                for order in open_orders:
                    # Identificar órdenes SHORT (SELL para abrir, BUY para cerrar con reduceOnly)
                    is_short_open = (order['side'] == 'SELL' and 
                                    order.get('reduceOnly', False) == False and
                                    order['type'] in ['LIMIT'])
                    
                    is_short_close = (order['side'] == 'BUY' and 
                                    order.get('reduceOnly', False) == True and
                                    order['type'] in ['LIMIT'])
                    
                    if is_short_open or is_short_close:
                        short_orders_found += 1
                        try:
                            cancel_result = self.client.futures_cancel_order(
                                symbol=symbol,
                                orderId=order['orderId'],
                                timestamp=int(time.time() * 1000)
                            )
                            cancelled_orders.append({
                                'orderId': order['orderId'],
                                'side': order['side'],
                                'type': order['type'],
                                'price': order['price'],
                                'quantity': order['origQty'],
                                'result': 'CANCELLED'
                            })
                            self.logger.info(f"Orden SHORT cancelada: {order['side']} {symbol} @ ${order['price']}")
                        except Exception as e:
                            cancelled_orders.append({
                                'orderId': order['orderId'],
                                'result': 'ERROR',
                                'error': str(e)
                            })
                
                if short_orders_found == 0:
                    self.logger.info(f"No se encontraron órdenes SHORT para {symbol}")
                    return None
                
                summary = {
                    'symbol': symbol,
                    'total_short_orders_found': short_orders_found,
                    'cancelled_orders': cancelled_orders,
                    'successful_cancellations': len([o for o in cancelled_orders if o['result'] == 'CANCELLED'])
                }
                
                self.logger.info(f"Cancelación SHORT completada: {symbol} | "
                                f"{summary['successful_cancellations']}/{short_orders_found} órdenes canceladas")
                
                return summary
                
            except Exception as e:
                self.logger.error(f"Error cancelando órdenes SHORT para {symbol}: {e}")
                return None

        def cancel_all_limit_orders(self, symbol: str) -> Optional[Dict]:
            """
            Cancela TODAS las órdenes LIMIT del símbolo (LONG y SHORT)
            
            Args:
                symbol: Par de trading
            
            Returns:
                Dict con resumen de cancelaciones
            """
            try:
                self.logger.info(f"Iniciando cancelación de todas las órdenes LIMIT para {symbol}")
                
                # Cancelar órdenes LONG
                long_result = self.cancel_limit_long(symbol)
                
                # Cancelar órdenes SHORT  
                short_result = self.cancel_limit_short(symbol)
                
                # Compilar resumen
                summary = {
                    'symbol': symbol,
                    'long_cancellations': long_result,
                    'short_cancellations': short_result,
                    'total_cancelled': 0
                }
                
                if long_result:
                    summary['total_cancelled'] += long_result.get('successful_cancellations', 0)
                
                if short_result:
                    summary['total_cancelled'] += short_result.get('successful_cancellations', 0)
                
                self.logger.info(f"Cancelación completa: {symbol} | "
                                f"{summary['total_cancelled']} órdenes LIMIT canceladas")
                
                return summary
                
            except Exception as e:
                self.logger.error(f"Error cancelando todas las órdenes LIMIT para {symbol}: {e}")
                return None
        
        def cancel_all_tp_sl_orders(self, symbol: str) -> Optional[Dict]:
            """
            Cancela todas las órdenes TAKE_PROFIT_MARKET y STOP_MARKET activas para un símbolo.
            
            Args:
                symbol: Par de trading (ej. BTCUSDT)
                
            Returns:
                Resumen de cancelaciones
            """
            try:
                open_orders = self.get_open_orders(symbol)
                if not open_orders:
                    self.logger.info(f"No hay órdenes abiertas para {symbol}")
                    return None
                
                cancelled_orders = []
                tp_sl_orders_found = 0
                
                for order in open_orders:
                    if order['type'] in ['TAKE_PROFIT', 'TAKE_PROFIT_MARKET', 'STOP', 'STOP_MARKET']:
                        tp_sl_orders_found += 1
                        try:
                            cancel_result = self.client.futures_cancel_order(
                                symbol=symbol,
                                orderId=order['orderId'],
                                timestamp=int(time.time() * 1000)
                            )
                            cancelled_orders.append({
                                'orderId': order['orderId'],
                                'type': order['type'],
                                'result': 'CANCELLED'
                            })
                            self.logger.info(f"Orden TP/SL cancelada: {order['type']} {symbol}")
                        except Exception as e:
                            cancelled_orders.append({
                                'orderId': order['orderId'],
                                'result': 'ERROR',
                                'error': str(e)
                            })
                
                summary = {
                    'symbol': symbol,
                    'total_tp_sl_orders_found': tp_sl_orders_found,
                    'cancelled_orders': cancelled_orders,
                    'successful_cancellations': len([o for o in cancelled_orders if o['result'] == 'CANCELLED'])
                }
                
                self.logger.info(f"Cancelación TP/SL completada: {symbol} | "
                                f"{summary['successful_cancellations']}/{tp_sl_orders_found} órdenes canceladas")
                
                return summary
            
            except Exception as e:
                self.logger.error(f"Error cancelando órdenes TP/SL para {symbol}: {e}")
                return None


       