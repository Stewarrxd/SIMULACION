# binance_api.py
from binance.client import Client
from binance.enums import *
import pandas as pd
import time

class BinanceAPI:
    def __init__(self, api_key, api_secret):
        if not api_key or not api_secret:
            raise ValueError("API Key y API Secret son requeridos")
        print(f"Inicializando API con Key: {api_key[:5]}...")
        self.client = Client(api_key, api_secret, testnet=False)
        
    def get_historical_klines(self, symbol, interval, limit=100):
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            return klines
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return None

    def get_position_info(self, symbol):
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            for position in positions:
                if position['symbol'] == symbol:
                    return position
            return None
        except Exception as e:
            print(f"Error getting position info: {e}")
            return None

    def open_position(self, symbol, side, quantity,DIRECCION):
        try:
            timestamp = int(time.time() * 1000)
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity,
                positionSide=DIRECCION,
                timestamp = int(time.time() * 1000)
                
            )
            return order
        except Exception as e:
            print(f"Error opening position: {e}")
            return None

    def close_position(self, symbol, side, quantity,DIRECCION):
        try:
            order = self.client.futures_create_order(
               symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity,
                positionSide=DIRECCION,
                timestamp = int(time.time() * 1000)
            )
            return order
        except Exception as e:
            print(f"Error closing position: {e}")
            return None
    
    def create_order(self, symbol, side, order_type, quantity, positionSide, callbackRate, reduceOnly):
        try:
            order_params = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": quantity,
                "timestamp": int(time.time() * 1000),
                "positionSide": positionSide
            }
            if callbackRate is not None:
                order_params["callbackRate"] = callbackRate
            if reduceOnly:
                hola:6

            
            order = self.client.futures_create_order(**order_params)
            return order
        except Exception as e:
            print(f"Error creating order: {e}")
            return None
        
    def create_order_trailing(self, symbol, side, quantity, positionSide, callbackRate, activationPrice=None, reduceOnly=True):
        try:
            order_params = {
                "symbol": symbol,
                "side": side,
                "type": 'LIMIT',
                "quantity": quantity,
                "timestamp": int(time.time() * 1000),
                "positionSide": positionSide,
                "timeInForce": "GTC"  # Good Till Canceled
                
                
            }
            if activationPrice is not None:
                order_params["price"] = round(activationPrice,4)
            order = self.client.futures_create_order(**order_params)
            return order
        except Exception as e:
            print(f"Error creating trailing stop market order: {e}")
            return None


    def cancel_order(self, symbol, orderId):
        try:
            order = self.client.futures_cancel_order(
                symbol=symbol,
                orderId=orderId,
                timestamp=int(time.time() * 1000)
            )
            return order
        except Exception as e:
            print(f"Error cancelling order: {e}")
            return None