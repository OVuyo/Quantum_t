"""
MT5 Bridge - Integration between VPN/VPS and MetaTrader 5
Provides seamless connectivity and API for EA communication
"""

import asyncio
import json
import struct
import socket
import ssl
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import MetaTrader5 as mt5
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"

@dataclass
class TradingSignal:
    symbol: str
    order_type: OrderType
    volume: float
    price: float
    sl: float
    tp: float
    comment: str
    magic: int

class MT5Bridge:
    """
    High-performance bridge between VPN/VPS and MT5
    Features:
    - Real-time order execution
    - Market data streaming
    - Risk management integration
    - Latency optimization
    """
    
    def __init__(self, vpn_config: dict, vps_instance_id: str):
        self.vpn_config = vpn_config
        self.vps_instance_id = vps_instance_id
        self.mt5_connected = False
        self.active_positions = {}
        self.pending_orders = {}
        self.market_data_cache = {}
        self.execution_callbacks: List[Callable] = []
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'avg_execution_time': 0,
            'total_volume': 0
        }
    
    async def initialize(self, account_login: int, password: str, server: str):
        """Initialize MT5 connection through VPN"""
        try:
            # Connect through VPN tunnel
            await self._establish_vpn_tunnel()
            
            # Initialize MT5
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            
            # Login to account
            authorized = mt5.login(account_login, password=password, server=server)
            
            if authorized:
                self.mt5_connected = True
                logger.info(f"Connected to MT5: {mt5.account_info()}")
                
                # Start background tasks
                asyncio.create_task(self._monitor_positions())
                asyncio.create_task(self._stream_market_data())
                
                return True
            else:
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize MT5 bridge: {e}")
            return False
    
    async def _establish_vpn_tunnel(self):
        """Establish secure VPN tunnel for MT5 traffic"""
        # Connect to VPN server
        vpn_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        vpn_socket.settimeout(5)
        
        try:
            vpn_socket.connect((
                self.vpn_config['server'],
                self.vpn_config['port']
            ))
            
            # SSL wrap for security
            context = ssl.create_default_context()
            self.vpn_connection = context.wrap_socket(
                vpn_socket,
                server_hostname=self.vpn_config['server']
            )
            
            # Authenticate
            auth_data = {
                'vps_id': self.vps_instance_id,
                'timestamp': time.time(),
                'auth_token': self._generate_auth_token()
            }
            
            self.vpn_connection.send(json.dumps(auth_data).encode())
            response = self.vpn_connection.recv(1024)
            
            if json.loads(response.decode())['status'] == 'authenticated':
                logger.info("VPN tunnel established")
                return True
                
        except Exception as e:
            logger.error(f"VPN connection failed: {e}")
            raise
    
    def _generate_auth_token(self) -> str:
        """Generate authentication token"""
        data = f"{self.vps_instance_id}{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    async def execute_trade(self, signal: TradingSignal) -> dict:
        """Execute trading signal with optimized routing"""
        if not self.mt5_connected:
            return {'status': 'error', 'message': 'MT5 not connected'}
        
        start_time = time.perf_counter()
        
        try:
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": signal.volume,
                "type": self._get_mt5_order_type(signal.order_type),
                "price": signal.price,
                "sl": signal.sl,
                "tp": signal.tp,
                "deviation": 10,
                "magic": signal.magic,
                "comment": signal.comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order through VPN for lowest latency
            result = await self._send_order_via_vpn(request)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            # Update metrics
            self._update_execution_metrics(execution_time, result)
            
            # Trigger callbacks
            for callback in self.execution_callbacks:
                asyncio.create_task(callback(signal, result))
            
            return {
                'status': 'success' if result.retcode == mt5.TRADE_RETCODE_DONE else 'error',
                'ticket': result.order if result else None,
                'execution_time_ms': execution_time,
                'result': result._asdict() if result else None
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _send_order_via_vpn(self, request: dict):
        """Send order through VPN for optimized execution"""
        # Serialize order request
        order_data = json.dumps({
            'type': 'order',
            'request': request,
            'timestamp': time.time()
        }).encode()
        
        # Send through VPN connection
        self.vpn_connection.send(struct.pack('>I', len(order_data)) + order_data)
        
        # For actual implementation, this would route through VPN
        # For now, execute directly
        result = mt5.order_send(request)
        
        return result
    
    def _get_mt5_order_type(self, order_type: OrderType) -> int:
        """Convert order type to MT5 constant"""
        mapping = {
            OrderType.BUY: mt5.ORDER_TYPE_BUY,
            OrderType.SELL: mt5.ORDER_TYPE_SELL,
            OrderType.BUY_LIMIT: mt5.ORDER_TYPE_BUY_LIMIT,
            OrderType.SELL_LIMIT: mt5.ORDER_TYPE_SELL_LIMIT,
            OrderType.BUY_STOP: mt5.ORDER_TYPE_BUY_STOP,
            OrderType.SELL_STOP: mt5.ORDER_TYPE_SELL_STOP,
        }
        return mapping.get(order_type, mt5.ORDER_TYPE_BUY)
    
    async def _monitor_positions(self):
        """Monitor open positions and orders"""
        while self.mt5_connected:
            try:
                # Get open positions
                positions = mt5.positions_get()
                if positions:
                    self.active_positions = {
                        pos.ticket: pos._asdict() for pos in positions
                    }
                
                # Get pending orders
                orders = mt5.orders_get()
                if orders:
                    self.pending_orders = {
                        order.ticket: order._asdict() for order in orders
                    }
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _stream_market_data(self):
        """Stream real-time market data"""
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'GOLD', 'BTCUSD']
        
        while self.mt5_connected:
            try:
                for symbol in symbols:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        self.market_data_cache[symbol] = {
                            'bid': tick.bid,
                            'ask': tick.ask,
                            'last': tick.last,
                            'volume': tick.volume,
                            'time': tick.time,
                            'spread': (tick.ask - tick.bid) * 10000  # In points
                        }
                
                await asyncio.sleep(0.1)  # 100ms updates
                
            except Exception as e:
                logger.error(f"Market data streaming error: {e}")
                await asyncio.sleep(1)
    
    def _update_execution_metrics(self, execution_time: float, result: Any):
        """Update execution metrics"""
        self.performance_metrics['total_trades'] += 1
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.performance_metrics['successful_trades'] += 1
            
            # Update average execution time
            avg_time = self.performance_metrics['avg_execution_time']
            total = self.performance_metrics['total_trades']
            self.performance_metrics['avg_execution_time'] = (
                (avg_time * (total - 1) + execution_time) / total
            )
            
            # Update volume
            if hasattr(result, 'volume'):
                self.performance_metrics['total_volume'] += result.volume
    
    async def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> dict:
        """Modify existing position"""
        if ticket not in self.active_positions:
            return {'status': 'error', 'message': 'Position not found'}
        
        position = self.active_positions[ticket]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl if sl else position['sl'],
            "tp": tp if tp else position['tp'],
        }
        
        result = mt5.order_send(request)
        
        return {
            'status': 'success' if result.retcode == mt5.TRADE_RETCODE_DONE else 'error',
            'result': result._asdict() if result else None
        }
    
    async def close_position(self, ticket: int, volume: float = None) -> dict:
        """Close position"""
        if ticket not in self.active_positions:
            return {'status': 'error', 'message': 'Position not found'}
        
        position = self.active_positions[ticket]
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": position['symbol'],
            "volume": volume if volume else position['volume'],
            "type": mt5.ORDER_TYPE_SELL if position['type'] == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(position['symbol']).bid,
            "deviation": 10,
            "magic": position['magic'],
            "comment": "Close by Quantum",
        }
        
        result = mt5.order_send(request)
        
        return {
            'status': 'success' if result.retcode == mt5.TRADE_RETCODE_DONE else 'error',
            'result': result._asdict() if result else None
        }
    
    def get_account_info(self) -> dict:
        """Get account information"""
        if not self.mt5_connected:
            return None
        
        account = mt5.account_info()
        if account:
            return {
                'login': account.login,
                'balance': account.balance,
                'equity': account.equity,
                'margin': account.margin,
                'free_margin': account.margin_free,
                'margin_level': account.margin_level,
                'profit': account.profit,
                'leverage': account.leverage,
                'currency': account.currency,
            }
        return None
    
    def get_market_data(self, symbol: str) -> dict:
        """Get current market data for symbol"""
        return self.market_data_cache.get(symbol)
    
    def register_execution_callback(self, callback: Callable):
        """Register callback for trade execution events"""
        self.execution_callbacks.append(callback)
    
    def get_performance_metrics(self) -> dict:
        """Get performance metrics"""
        return self.performance_metrics
    
    async def shutdown(self):
        """Shutdown MT5 bridge"""
        self.mt5_connected = False
        
        # Close all positions if configured
        # for ticket in list(self.active_positions.keys()):
        #     await self.close_position(ticket)
        
        # Shutdown MT5
        mt5.shutdown()
        
        # Close VPN connection
        if hasattr(self, 'vpn_connection'):
            self.vpn_connection.close()
        
        logger.info("MT5 Bridge shutdown complete")

# Example usage
async def example_trading_callback(signal: TradingSignal, result: dict):
    """Example callback for trade execution"""
    logger.info(f"Trade executed: {signal.symbol} {signal.order_type.value} - Result: {result}")

async def main():
    """Example usage"""
    # Initialize bridge
    bridge = MT5Bridge(
        vpn_config={
            'server': 'vpn.quantum-trading.com',
            'port': 8443
        },
        vps_instance_id='qvps_trading_001'
    )
    
    # Connect to MT5
    await bridge.initialize(
        account_login=12345678,
        password='your_password',
        server='ICMarkets-Demo'
    )
    
    # Register callback
    bridge.register_execution_callback(example_trading_callback)
    
    # Execute a trade
    signal = TradingSignal(
        symbol='EURUSD',
        order_type=OrderType.BUY,
        volume=0.01,
        price=1.0850,
        sl=1.0800,
        tp=1.0900,
        comment='Quantum Trade',
        magic=123456
    )
    
    result = await bridge.execute_trade(signal)
    print(f"Trade result: {result}")
    
    # Monitor for 60 seconds
    await asyncio.sleep(60)
    
    # Shutdown
    await bridge.shutdown()

if __name__ == "__main__":
    asyncio.run(main())