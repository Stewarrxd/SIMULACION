# binance_fixed.py - Versión corregida para errores de position side
from binance.client import Client
from binance.enums import *
import pandas as pd
import time
import logging

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
            Coloca una orden de take profit utilizando una orden condicional de mercado.

            Se calcula la orientación de la orden a partir de ``position_side``:
            - Para una posición LONG, la toma de ganancias se logra vendiendo. El ``stopPrice`` debe estar por
              encima del precio de mercado actual, de modo que la orden sólo se active cuando el precio
              alcance o supere el nivel de take profit.
            - Para una posición SHORT, la toma de ganancias se logra comprando. El ``stopPrice`` debe estar
              por debajo del precio de mercado actual, de modo que la orden sólo se active cuando el precio
              caiga hasta el nivel de take profit.

            Si ``position_side`` no se especifica o es ``BOTH``, se asume que la posición es LONG para los
            cálculos de stopPrice (esto es coherente con el modo one-way de Binance, donde ``positionSide``
            siempre es ``BOTH``). La orden se envía siempre con ``positionSide='BOTH'`` para cumplir el modo
            one-way, pero la lógica interna determina la dirección correcta para ``side``.
            """
            try:
                # Obtener el precio de mercado actual y el tickSize
                ticker = self.get_ticker_price(symbol)
                mark_price = float(ticker['price'])

                symbol_info = self._get_symbol_info(symbol)
                tick_size = next(
                    float(f['tickSize'])
                    for f in symbol_info.get('filters', [])
                    if f.get('filterType') == 'PRICE_FILTER'
                )

                # Pequeño margen para evitar activación inmediata
                buffer = max(mark_price * 0.00001, tick_size * 10)

                # Determinar orientación (LONG por defecto)
                direction = (position_side or "LONG").upper()

                # Ajustar el precio de take profit según dirección y precio de mercado
                if direction == "LONG":
                    # Para LONG se vende. stopPrice debe estar por encima del precio actual
                    target_price = max(take_profit_price, mark_price + buffer)
                    side = "SELL"
                elif direction == "SHORT":
                    # Para SHORT se compra. stopPrice debe estar por debajo del precio actual
                    target_price = min(take_profit_price, mark_price - buffer)
                    side = "BUY"
                else:
                    # En caso de valor inesperado, asumir LONG
                    target_price = max(take_profit_price, mark_price + buffer)
                    side = "SELL"

                # Redondear el stopPrice al tick size
                stop_price = self._round_price_limit(symbol, target_price)

                # Determinar cantidad actual de la posición (en modo one-way siempre 'BOTH')
                qty = self._get_position_quantity(symbol, "BOTH")
                if qty <= 0:
                    self.logger.warning(f"No hay posición abierta para colocar TP en {symbol}")
                    return None

                order_params = {
                    'symbol':      symbol,
                    'side':        side,
                    'type':        'TAKE_PROFIT_MARKET',
                    'stopPrice':   stop_price,
                    'quantity':    qty,
                    'reduceOnly':  'true',
                    'workingType': 'MARK_PRICE',
                    'priceProtect':'true',
                    'positionSide': 'BOTH',
                    'timestamp':   int(time.time() * 1000)
                }

                order = self.client.futures_create_order(**order_params)
                order_id = order.get('orderId') if order else None
                self.logger.info(
                    f"TP MARKET colocado en {stop_price} para {symbol} "
                    f"(side: {side}, positionSide: BOTH, orderId: {order_id})"
                )
                return order

            except Exception as e:
                self.logger.error(f"Error placing take profit: {e}")
                return None

        def set_stop_loss(self, symbol: str, stop_price: float, position_side: Optional[str] = None) -> Optional[Dict]:
            """
            Coloca una orden de stop loss utilizando una orden condicional de mercado.

            Para LONG se vende cuando el precio cae hasta el stop loss; por lo tanto, ``stopPrice`` debe
            estar por debajo del precio de mercado actual. Para SHORT se compra cuando el precio sube
            hasta el stop loss; por lo tanto, ``stopPrice`` debe estar por encima del precio de mercado.

            Si ``position_side`` no se especifica o es ``BOTH``, se asume que la posición es LONG para los
            cálculos, pero la orden siempre se envía con ``positionSide='BOTH'`` en modo one-way.
            """
            try:
                # Obtener el precio de mercado actual y el tickSize
                ticker = self.get_ticker_price(symbol)
                mark_price = float(ticker['price'])

                symbol_info = self._get_symbol_info(symbol)
                tick_size = next(
                    float(f['tickSize'])
                    for f in symbol_info.get('filters', [])
                    if f.get('filterType') == 'PRICE_FILTER'
                )

                # Pequeño margen para evitar activación inmediata
                buffer = max(mark_price * 0.00001, tick_size * 10)

                # Determinar orientación (LONG por defecto)
                direction = (position_side or "LONG").upper()

                # Ajustar el stop_loss según dirección y precio de mercado
                if direction == "LONG":
                    # Para LONG se vende cuando el precio cae: stopPrice por debajo del mercado
                    target_price = min(stop_price, mark_price - buffer)
                    side = "SELL"
                elif direction == "SHORT":
                    # Para SHORT se compra cuando el precio sube: stopPrice por encima del mercado
                    target_price = max(stop_price, mark_price + buffer)
                    side = "BUY"
                else:
                    target_price = min(stop_price, mark_price - buffer)
                    side = "SELL"

                # Redondear el stopPrice al tick size
                stop_price_rounded = self._round_price_limit(symbol, target_price)

                # Obtener cantidad de la posición en modo one-way
                qty = self._get_position_quantity(symbol, "BOTH")
                if qty <= 0:
                    self.logger.warning(f"No hay posición abierta para colocar SL en {symbol}")
                    return None

                order_params = {
                    'symbol':      symbol,
                    'side':        side,
                    'type':        'STOP_MARKET',
                    'stopPrice':   stop_price_rounded,
                    'quantity':    qty,
                    'reduceOnly':  'true',
                    'workingType': 'MARK_PRICE',
                    'priceProtect':'true',
                    'positionSide': 'BOTH',
                    'timestamp':   int(time.time() * 1000)
                }

                order = self.client.futures_create_order(**order_params)
                order_id = order.get('orderId') if order else None
                self.logger.info(
                    f"SL MARKET colocado en {stop_price_rounded} para {symbol} "
                    f"(side: {side}, positionSide: BOTH, orderId: {order_id})"
                )
                return order

            except Exception as e:
                self.logger.error(f"Error placing stop loss: {e}")
                return None

        
        
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


                # ============ BATCH ORDERS / BRACKET EN 1 REQUEST ============
        def _normalize_batch_order(self, symbol: str, order: dict) -> dict:
            """
            Normaliza una sub-orden para batch:
            - Rellena/valida positionSide con _get_correct_position_side
            - Redondea quantity, price, stopPrice
            - Asegura newClientOrderId <= 36 chars
            """
            import time, json, math
            o = order.copy()

            # symbol puede venir en cada orden; si no, usamos el que nos pasan arriba
            o.setdefault('symbol', symbol)

            # positionSide automático si no lo especifican
            side = o.get('side')
            if side:
                forced = o.get('positionSide')  # 'BOTH' | 'LONG' | 'SHORT' (opcional)
                o['positionSide'] = self._get_correct_position_side(side, forced)

            # timeInForce por defecto para órdenes con price
            if o.get('type') in ('LIMIT', 'STOP', 'TAKE_PROFIT') and 'timeInForce' not in o:
                o['timeInForce'] = 'GTC'

            # Redondeos
            if 'quantity' in o:
                o['quantity'] = self._round_quantity(o['symbol'], float(o['quantity']))
            if 'price' in o:
                o['price'] = self._round_price(o['symbol'], float(o['price']))
            if 'stopPrice' in o:
                # Para triggers usamos el filtro de tickSize exacto
                o['stopPrice'] = self._round_price_limit(o['symbol'], float(o['stopPrice']))

            # Idempotencia por sub-orden (máx 36 chars)
            if 'newClientOrderId' not in o:
                o['newClientOrderId'] = self._generate_order_id(o['type'], o['symbol'])
            else:
                o['newClientOrderId'] = str(o['newClientOrderId'])[:36]

            # reduceOnly debe ser boolean/bool-compatible
            if 'reduceOnly' in o:
                o['reduceOnly'] = bool(o['reduceOnly'])

            return o


        def place_batch_orders(self, orders: list, symbol: str = None, max_retries: int = 1):
            """
            Envía varias órdenes en 1 request al endpoint batch de Binance Futures.
            - 'orders' es una lista de dicts con los mismos campos que futures_create_order
            - Opcionalmente pasa 'symbol' para normalizar redondeos al mismo par
            Devuelve la lista de resultados por sub-orden (en el mismo orden).
            """
            import json, time

            if not isinstance(orders, list) or len(orders) == 0:
                raise ValueError("orders debe ser una lista con al menos 1 elemento")
            if len(orders) > 5:
                raise ValueError("Binance Futures batchOrders acepta máximo 5 sub-órdenes")

            # Normalización previa para minimizar rechazos de PRICE_FILTER/stepSize
            norm = [self._normalize_batch_order(o.get('symbol', symbol) or orders[0].get('symbol'), o)
                    for o in orders]

            payload = {'batchOrders': json.dumps(norm), 'timestamp': int(time.time() * 1000)}

            # Llamamos al método de la librería si existe; si no, usamos la ruta interna
            for attempt in range(1, max_retries + 1):
                try:
                    if hasattr(self.client, 'futures_place_batch_order'):
                        resp = self.client.futures_place_batch_order(batchOrders=norm)
                    else:
                        # Fallback al request interno si la versión de python-binance no expone el helper
                        resp = self.client._request_futures_api('post', 'batchOrders', True, data=payload)
                    # Actualiza cache TP/SL si procede
                    try:
                        for r in resp or []:
                            typ = str(r.get('type', '')).upper()
                            sym = r.get('symbol')
                            if typ in ('TAKE_PROFIT', 'TAKE_PROFIT_MARKET', 'STOP', 'STOP_MARKET') and sym:
                                self.tp_sl_orders.setdefault(sym, {'tp': None, 'sl': None})
                                if 'TAKE_PROFIT' in typ:
                                    self.tp_sl_orders[sym]['tp'] = r.get('orderId')
                                if 'STOP' in typ:
                                    self.tp_sl_orders[sym]['sl'] = r.get('orderId')
                    except Exception:
                        pass
                    return resp
                except Exception as e:
                    self.logger.warning(f"Batch intento {attempt} falló: {e}")
                    if attempt >= max_retries:
                        raise

        def bracket_batch(self, symbol: str, side: str, quantity: float,
                        entry_type: str = 'MARKET',
                        entry_price: float = None,
                        take_profit: float = None,
                        stop_loss: float = None,
                        time_in_force: str = 'GTC',
                        position_side: str = None):
            """
            Envía EN 1 REQUEST:
            - Orden de entrada (MARKET o LIMIT)
            - TP (TAKE_PROFIT LIMIT) reduceOnly
            - SL (STOP LIMIT) reduceOnly
            """
            side = side.upper()
            assert side in ('BUY', 'SELL')
            qty = float(quantity)

            orders = []

            # 1) Entrada
            entry = {
                'symbol': symbol,
                'side': side,
                'type': entry_type.upper(),
                'quantity': qty,
                'positionSide': self._get_correct_position_side(side, position_side),
            }
            if entry['type'] == 'LIMIT':
                if entry_price is None:
                    raise ValueError("entry_price es requerido para LIMIT")
                entry['price'] = entry_price
                entry['timeInForce'] = time_in_force
            orders.append(entry)

            # 2) TP (opcional)
            if take_profit is not None:
                # Para LONG vendemos; para SHORT compramos
                tp_side = 'SELL' if side == 'BUY' else 'BUY'
                tp_sp = float(take_profit)
                tp_price = tp_sp  # usamos limit = stopPrice para precisión
                orders.append({
                    'symbol': symbol,
                    'side': tp_side,
                    'type': 'TAKE_PROFIT',
                    'stopPrice': tp_sp,
                    'price': tp_price,
                    'quantity': qty,
                    'timeInForce': 'GTC',
                    'reduceOnly': True,
                    'positionSide': self._get_correct_position_side(tp_side, position_side if position_side in ('LONG','SHORT') else 'BOTH')
                })

            # 3) SL (opcional)
            if stop_loss is not None:
                sl_side = 'SELL' if side == 'BUY' else 'BUY'
                sl_sp = float(stop_loss)
                # límite ligeramente peor para asegurar fill al disparar
                limit_worse = sl_sp * (0.9995 if sl_side == 'SELL' else 1.0005)
                orders.append({
                    'symbol': symbol,
                    'side': sl_side,
                    'type': 'STOP',
                    'stopPrice': sl_sp,
                    'price': limit_worse,
                    'quantity': qty,
                    'timeInForce': 'GTC',
                    'reduceOnly': True,
                    'positionSide': self._get_correct_position_side(sl_side, position_side if position_side in ('LONG','SHORT') else 'BOTH')
                })

            return self.place_batch_orders(orders, symbol=symbol)
# ============ /BATCH ORDERS ==================================




        # ======================== EJEMPLO DE USO COMPLETO ========================

        """

        api = BinanceAPI(API_KEY, API_SECRET, testnet=False)

# 1) Entrada MARKET + TP + SL en 1 request (BUY/LONG)
api.bracket_batch(
    symbol="BTCUSDT",
    side="BUY",
    quantity=0.005,
    entry_type="MARKET",
    take_profit=72000.0,
    stop_loss=68000.0
)

# 2) Entrada LIMIT + TP + SL en 1 request (SELL/SHORT)
api.bracket_batch(
    symbol="ETHUSDT",
    side="SELL",
    quantity=0.2,
    entry_type="LIMIT",
    entry_price=3550.0,
    take_profit=3300.0,
    stop_loss=3650.0
)






        # Para usar estas funciones, agrégalas a tu clase BinanceAPI y úsalas así:

        api = BinanceAPI(API_KEY, API_SECRET, testnet=False)

        # 1. ABRIR POSICIONES LIMIT
        long_order = api.limit_open_long("BTCUSDT", 0.01, 45000.0, leverage=50)
        short_order = api.limit_open_short("ETHUSDT", 0.1, 3500.0, leverage=50)

        # Guardar los IDs de las órdenes
        long_order_id = long_order['orderId'] if long_order else None
        short_order_id = short_order['orderId'] if short_order else None

        # 2. CERRAR POSICIONES LIMIT (cuando tengas posiciones abiertas)
        # Opción A: Especificar todo manualmente
        exit_long = api.limit_exit_long("BTCUSDT", 0.01, 46000.0)

        # Opción B: Cerrar toda la posición automáticamente
        exit_short = api.limit_exit_short("ETHUSDT")  # Detecta cantidad y precio automáticamente

        # Opción C: Solo especificar precio, detectar cantidad
        exit_long2 = api.limit_exit_long("BTCUSDT", limit_price=47000.0)

        # 3. CANCELAR ÓRDENES ESPECÍFICAS
        # Cancelar una orden específica por ID
        api.cancel_limit_long("BTCUSDT", long_order_id)
        api.cancel_limit_short("ETHUSDT", short_order_id)

        # 4. CANCELAR TODAS LAS ÓRDENES DE UN TIPO
        # Cancelar todas las órdenes LONG del símbolo
        api.cancel_limit_long("BTCUSDT")

        # Cancelar todas las órdenes SHORT del símbolo  
        api.cancel_limit_short("ETHUSDT")

        # 5. CANCELAR TODO
        # Cancelar todas las órdenes LIMIT (LONG y SHORT) del símbolo
        api.cancel_all_limit_orders("BTCUSDT")

        # EJEMPLO DE FLUJO COMPLETO:
        # Abrir → Esperar → Cancelar si no se ejecuta
        long_order = api.limit_open_long("BTCUSDT", 0.01, 45000.0, leverage=50)
        if long_order:
            order_id = long_order['orderId']
            
            # Esperar 5 minutos
            time.sleep(300)
            
            # Verificar si se ejecutó, si no, cancelar
            order_status = api.check_order_status("BTCUSDT", order_id)
            if order_status and order_status['status'] not in ['FILLED', 'PARTIALLY_FILLED']:
                print("Orden no ejecutada, cancelando...")
                api.cancel_limit_long("BTCUSDT", order_id)
        """


# ======================== EJEMPLO DE USO ========================

if __name__ == "__main__":
    # Configuración
    API_KEY = "bfTHsxAHNKQsSpW9u9SmIUBM1EnDQX8eAvXUyxlsTQDKEpndcnxkaY9PIpceD9o2"
    API_SECRET = "Su9OdJzTR8x0nXlC1xUNU9AXtGxu4jtVUod8bZmbGtY3iKdHKUW78YkJMbw4UqAQ"
    
    # Inicializar API (usar testnet=True para pruebas)
    api = BinanceAPI(API_KEY, API_SECRET, testnet=False)
    
    symbol = "BTCUSDT"
    
    try:
        # Configurar apalancamiento
        api.set_leverage(symbol, 125)
        
        # Configurar margen cruzado
        api.set_margin_type(symbol, "CROSSED")
        
        # Obtener precio actual
        ticker = api.get_ticker_price(symbol)
        current_price = float(ticker['price'])
        print(f"Precio actual de {symbol}: ${current_price}")
        
        # Abrir posición larga
        quantity = 0.01
        order = api.open_long_position(symbol, quantity)
        if order:
            print(f"Posición larga abierta: {order}")
            
            # Establecer stop loss 2% por debajo
            stop_price = current_price * 0.98
            sl_order = api.set_stop_loss(symbol, stop_price, "LONG")
            print(f"Stop loss establecido: {sl_order}")
            
            # Establecer take profit 3% por encima
            tp_price = current_price * 1.03
            tp_order = api.set_take_profit(symbol, tp_price, "LONG")
            print(f"Take profit establecido: {tp_order}")
        
        # Verificar posiciones
        positions = api.get_position_info(symbol)
        print(f"Información de posición: {positions}")
        
        # Verificar órdenes abiertas
        open_orders = api.get_open_orders(symbol)
        print(f"Órdenes abiertas: {open_orders}")
        
    except Exception as e:
        print(f"Error en el ejemplo: {e}")