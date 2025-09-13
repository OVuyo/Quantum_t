"""
Quantum Trading REST API
Provides HTTP endpoints for managing VPN/VPS infrastructure
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import time
from datetime import datetime
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.vps.vps_manager import QuantumVPS, VPSInstance
from infrastructure.vpn.vpn_server import QuantumVPN
from infrastructure.integration.mt5_bridge import MT5Bridge, TradingSignal, OrderType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quantum Trading API",
    description="REST API for Quantum Trading Infrastructure",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
vps_manager = QuantumVPS()
vpn_server = None
mt5_bridges: Dict[str, MT5Bridge] = {}

# Pydantic models
class VPSCreateRequest(BaseModel):
    name: str
    cpu_cores: Optional[int] = 2
    ram_mb: Optional[int] = 4096
    disk_gb: Optional[int] = 50

class MT5DeployRequest(BaseModel):
    instance_id: str
    account: Dict[str, Any]

class TradeRequest(BaseModel):
    instance_id: str
    symbol: str
    order_type: str
    volume: float
    price: Optional[float] = 0
    sl: Optional[float] = 0
    tp: Optional[float] = 0
    comment: Optional[str] = ""
    magic: Optional[int] = 0

class ScaleRequest(BaseModel):
    instance_id: str
    cpu_cores: Optional[int] = None
    ram_mb: Optional[int] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global vpn_server
    
    # Start VPN server in background
    vpn_server = QuantumVPN()
    asyncio.create_task(vpn_server.start_server())
    
    # Start VPS monitoring
    asyncio.create_task(vps_manager.monitor_resources())
    
    logger.info("Quantum Trading API started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    # Shutdown MT5 bridges
    for bridge in mt5_bridges.values():
        await bridge.shutdown()
    
    logger.info("Quantum Trading API shutdown")

# Health check endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Quantum Trading Infrastructure",
        "timestamp": datetime.now().isoformat()
    }

# VPS Endpoints
@app.post("/vps/create")
async def create_vps(request: VPSCreateRequest):
    """Create a new VPS instance"""
    try:
        instance = await vps_manager.create_instance(
            name=request.name,
            cpu_cores=request.cpu_cores,
            ram_mb=request.ram_mb,
            disk_gb=request.disk_gb
        )
        
        return {
            "status": "success",
            "instance_id": instance.instance_id,
            "details": {
                "name": instance.name,
                "cpu_cores": instance.cpu_cores,
                "ram_mb": instance.ram_mb,
                "disk_gb": instance.disk_gb,
                "status": instance.status
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vps/list")
async def list_vps():
    """List all VPS instances"""
    instances = []
    for instance_id, instance in vps_manager.instances.items():
        instances.append({
            "instance_id": instance_id,
            "name": instance.name,
            "status": instance.status,
            "cpu_cores": instance.cpu_cores,
            "ram_mb": instance.ram_mb,
            "mt5_terminals": len(instance.mt5_terminals),
            "created_at": instance.created_at.isoformat()
        })
    
    return {"instances": instances}

@app.get("/vps/{instance_id}")
async def get_vps(instance_id: str):
    """Get VPS instance details"""
    if instance_id not in vps_manager.instances:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    instance = vps_manager.instances[instance_id]
    
    # Get resource usage
    monitor_data = await vps_manager._get_resource_usage(instance)
    
    return {
        "instance_id": instance_id,
        "name": instance.name,
        "status": instance.status,
        "resources": {
            "cpu_cores": instance.cpu_cores,
            "ram_mb": instance.ram_mb,
            "disk_gb": instance.disk_gb
        },
        "usage": {
            "cpu_percent": monitor_data.cpu_percent,
            "memory_percent": monitor_data.memory_percent,
            "disk_usage": monitor_data.disk_usage
        },
        "mt5_terminals": instance.mt5_terminals,
        "performance_score": instance.performance_score
    }

@app.post("/vps/scale")
async def scale_vps(request: ScaleRequest):
    """Scale VPS instance resources"""
    try:
        await vps_manager.scale_instance(
            instance_id=request.instance_id,
            cpu_cores=request.cpu_cores,
            ram_mb=request.ram_mb
        )
        
        return {
            "status": "success",
            "message": f"Instance {request.instance_id} scaled successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/vps/{instance_id}")
async def delete_vps(instance_id: str):
    """Delete VPS instance"""
    if instance_id not in vps_manager.instances:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    # Shutdown MT5 bridge if exists
    if instance_id in mt5_bridges:
        await mt5_bridges[instance_id].shutdown()
        del mt5_bridges[instance_id]
    
    # Remove instance
    del vps_manager.instances[instance_id]
    
    return {"status": "success", "message": f"Instance {instance_id} deleted"}

# MT5 Endpoints
@app.post("/mt5/deploy")
async def deploy_mt5(request: MT5DeployRequest):
    """Deploy MT5 terminal on VPS"""
    try:
        terminal_id = await vps_manager.deploy_mt5_terminal(
            instance_id=request.instance_id,
            account_config=request.account
        )
        
        # Create MT5 bridge
        bridge = MT5Bridge(
            vpn_config={'server': 'localhost', 'port': 8443},
            vps_instance_id=request.instance_id
        )
        
        # Initialize MT5 connection
        success = await bridge.initialize(
            account_login=request.account['login'],
            password=request.account['password'],
            server=request.account['server']
        )
        
        if success:
            mt5_bridges[request.instance_id] = bridge
            
            return {
                "status": "success",
                "terminal_id": terminal_id,
                "instance_id": request.instance_id
            }
        else:
            raise HTTPException(status_code=500, detail="MT5 initialization failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trade/execute")
async def execute_trade(request: TradeRequest):
    """Execute trade on MT5"""
    if request.instance_id not in mt5_bridges:
        raise HTTPException(status_code=404, detail="MT5 bridge not found for instance")
    
    bridge = mt5_bridges[request.instance_id]
    
    # Create trading signal
    signal = TradingSignal(
        symbol=request.symbol,
        order_type=OrderType[request.order_type],
        volume=request.volume,
        price=request.price,
        sl=request.sl,
        tp=request.tp,
        comment=request.comment,
        magic=request.magic
    )
    
    # Execute trade
    result = await bridge.execute_trade(signal)
    
    return result

@app.get("/trade/positions/{instance_id}")
async def get_positions(instance_id: str):
    """Get open positions"""
    if instance_id not in mt5_bridges:
        raise HTTPException(status_code=404, detail="MT5 bridge not found")
    
    bridge = mt5_bridges[instance_id]
    return {"positions": bridge.active_positions}

@app.get("/trade/orders/{instance_id}")
async def get_orders(instance_id: str):
    """Get pending orders"""
    if instance_id not in mt5_bridges:
        raise HTTPException(status_code=404, detail="MT5 bridge not found")
    
    bridge = mt5_bridges[instance_id]
    return {"orders": bridge.pending_orders}

@app.post("/trade/close/{instance_id}/{ticket}")
async def close_position(instance_id: str, ticket: int):
    """Close position"""
    if instance_id not in mt5_bridges:
        raise HTTPException(status_code=404, detail="MT5 bridge not found")
    
    bridge = mt5_bridges[instance_id]
    result = await bridge.close_position(ticket)
    
    return result

# VPN Endpoints
@app.get("/vpn/status")
async def vpn_status():
    """Get VPN server status"""
    if not vpn_server:
        return {"status": "offline"}
    
    return {
        "status": "online",
        "active_connections": len(vpn_server.active_connections),
        "performance_metrics": vpn_server.performance_metrics,
        "trading_routes": len(vpn_server.trading_routes)
    }

@app.get("/vpn/routes")
async def get_vpn_routes():
    """Get VPN trading routes"""
    if not vpn_server:
        return {"routes": []}
    
    routes = []
    for broker, route in vpn_server.trading_routes.items():
        routes.append({
            "broker": broker,
            "server": route.broker_server,
            "latency_ms": route.latency_ms,
            "priority": route.priority
        })
    
    return {"routes": routes}

# Metrics Endpoints
@app.get("/metrics")
async def get_metrics():
    """Get overall system metrics"""
    metrics = {
        "vps": {
            "total_instances": len(vps_manager.instances),
            "active_instances": len([i for i in vps_manager.instances.values() if i.status == 'running']),
            "total_mt5_terminals": sum(len(i.mt5_terminals) for i in vps_manager.instances.values())
        },
        "vpn": vpn_server.performance_metrics if vpn_server else {},
        "mt5": {}
    }
    
    # Aggregate MT5 metrics
    total_trades = 0
    avg_latency = 0
    
    for bridge in mt5_bridges.values():
        bridge_metrics = bridge.get_performance_metrics()
        total_trades += bridge_metrics['total_trades']
        avg_latency += bridge_metrics['avg_execution_time']
    
    if mt5_bridges:
        avg_latency /= len(mt5_bridges)
    
    metrics["mt5"] = {
        "total_trades": total_trades,
        "avg_execution_time": avg_latency,
        "active_bridges": len(mt5_bridges)
    }
    
    return metrics

@app.get("/metrics/performance")
async def get_performance():
    """Get detailed performance metrics"""
    return {
        "performance_history": vps_manager.performance_history[-100:],  # Last 100 entries
        "report": vps_manager.get_performance_report()
    }

# Backup Endpoints
@app.post("/backup/{instance_id}")
async def create_backup(instance_id: str, background_tasks: BackgroundTasks):
    """Create backup of VPS instance"""
    if instance_id not in vps_manager.instances:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    backup_path = "/opt/quantum/backups"
    os.makedirs(backup_path, exist_ok=True)
    
    # Run backup in background
    background_tasks.add_task(
        vps_manager.backup_instance,
        instance_id,
        backup_path
    )
    
    return {
        "status": "success",
        "message": f"Backup started for instance {instance_id}"
    }

@app.post("/restore")
async def restore_backup(backup_file: str):
    """Restore VPS instance from backup"""
    try:
        instance = await vps_manager.restore_instance(backup_file)
        
        return {
            "status": "success",
            "instance_id": instance.instance_id,
            "message": "Instance restored successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)