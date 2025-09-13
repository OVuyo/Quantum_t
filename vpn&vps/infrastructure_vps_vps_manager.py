"""
Quantum Trading VPS Manager - Self-Contained Virtual Private Server
Optimized for MT5 EA execution with resource management and auto-scaling
"""

import os
import sys
import time
import json
import psutil
import docker
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import subprocess
import platform
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VPSInstance:
    """VPS Instance configuration"""
    instance_id: str
    name: str
    cpu_cores: int
    ram_mb: int
    disk_gb: int
    status: str
    created_at: datetime
    mt5_terminals: List[str]
    performance_score: float = 100.0

@dataclass
class ResourceMonitor:
    """Resource monitoring data"""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_latency: float
    active_trades: int
    
class QuantumVPS:
    """
    Self-contained VPS optimized for trading
    Features:
    - Auto-scaling based on trading load
    - Resource isolation for each EA
    - Automatic MT5 terminal management
    - Performance optimization
    - Crash recovery and persistence
    """
    
    def __init__(self, config_path: str = "vps_config.json"):
        self.config = self._load_config(config_path)
        self.instances: Dict[str, VPSInstance] = {}
        self.docker_client = None
        self.resource_monitor = None
        self.performance_history = []
        
        # Initialize Docker if available
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker environment initialized")
        except:
            logger.warning("Docker not available, using process isolation")
    
    def _load_config(self, config_path: str) -> dict:
        """Load VPS configuration"""
        default_config = {
            "max_instances": 10,
            "default_resources": {
                "cpu_cores": 2,
                "ram_mb": 4096,
                "disk_gb": 50
            },
            "auto_scaling": {
                "enabled": True,
                "cpu_threshold": 80,
                "memory_threshold": 85,
                "scale_up_cooldown": 300
            },
            "mt5_config": {
                "terminal_path": "/opt/mt5",
                "data_path": "/opt/mt5_data",
                "max_terminals_per_instance": 5
            },
            "monitoring": {
                "interval_seconds": 10,
                "history_retention_hours": 168
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return default_config
    
    async def create_instance(self, name: str, cpu_cores: int = None, 
                            ram_mb: int = None, disk_gb: int = None) -> VPSInstance:
        """Create a new VPS instance"""
        instance_id = f"qvps_{name}_{int(time.time())}"
        
        # Use provided resources or defaults
        cpu_cores = cpu_cores or self.config['default_resources']['cpu_cores']
        ram_mb = ram_mb or self.config['default_resources']['ram_mb']
        disk_gb = disk_gb or self.config['default_resources']['disk_gb']
        
        instance = VPSInstance(
            instance_id=instance_id,
            name=name,
            cpu_cores=cpu_cores,
            ram_mb=ram_mb,
            disk_gb=disk_gb,
            status='creating',
            created_at=datetime.now(),
            mt5_terminals=[]
        )
        
        self.instances[instance_id] = instance
        
        # Create isolated environment
        if self.docker_client:
            await self._create_docker_instance(instance)
        else:
            await self._create_process_instance(instance)
        
        instance.status = 'running'
        logger.info(f"Created VPS instance: {instance_id}")
        
        return instance
    
    async def _create_docker_instance(self, instance: VPSInstance):
        """Create Docker-based VPS instance"""
        try:
            # Create Docker container with resource limits
            container = self.docker_client.containers.run(
                "ubuntu:22.04",
                name=instance.instance_id,
                detach=True,
                mem_limit=f"{instance.ram_mb}m",
                cpu_quota=instance.cpu_cores * 100000,
                cpu_period=100000,
                volumes={
                    f"/opt/quantum_vps/{instance.instance_id}": {
                        'bind': '/data',
                        'mode': 'rw'
                    }
                },
                environment={
                    'VPS_ID': instance.instance_id,
                    'MT5_ENABLED': 'true'
                },
                restart_policy={"Name": "always"},
                command="tail -f /dev/null"  # Keep container running
            )
            
            # Install MT5 and dependencies in container
            setup_commands = [
                "apt-get update",
                "apt-get install -y wine64 xvfb",
                "mkdir -p /opt/mt5",
                # Add more MT5 setup commands
            ]
            
            for cmd in setup_commands:
                container.exec_run(cmd)
            
            logger.info(f"Docker container created: {container.id}")
            
        except Exception as e:
            logger.error(f"Failed to create Docker instance: {e}")
            raise
    
    async def _create_process_instance(self, instance: VPSInstance):
        """Create process-based VPS instance (fallback)"""
        # Create directory structure
        instance_dir = f"/opt/quantum_vps/{instance.instance_id}"
        os.makedirs(instance_dir, exist_ok=True)
        os.makedirs(f"{instance_dir}/mt5", exist_ok=True)
        os.makedirs(f"{instance_dir}/data", exist_ok=True)
        os.makedirs(f"{instance_dir}/logs", exist_ok=True)
        
        # Create instance configuration
        config = {
            'instance_id': instance.instance_id,
            'resources': {
                'cpu_cores': instance.cpu_cores,
                'ram_mb': instance.ram_mb,
                'disk_gb': instance.disk_gb
            },
            'created_at': instance.created_at.isoformat()
        }
        
        with open(f"{instance_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Process-based instance created: {instance_dir}")
    
    async def deploy_mt5_terminal(self, instance_id: str, 
                                 account_config: dict) -> str:
        """Deploy MT5 terminal on VPS instance"""
        if instance_id not in self.instances:
            raise ValueError(f"Instance {instance_id} not found")
        
        instance = self.instances[instance_id]
        terminal_id = f"mt5_{len(instance.mt5_terminals) + 1}"
        
        # Check resource availability
        if len(instance.mt5_terminals) >= self.config['mt5_config']['max_terminals_per_instance']:
            raise ValueError(f"Maximum terminals reached for instance {instance_id}")
        
        # Deploy terminal
        if self.docker_client:
            await self._deploy_mt5_docker(instance, terminal_id, account_config)
        else:
            await self._deploy_mt5_process(instance, terminal_id, account_config)
        
        instance.mt5_terminals.append(terminal_id)
        logger.info(f"Deployed MT5 terminal {terminal_id} on {instance_id}")
        
        return terminal_id
    
    async def _deploy_mt5_docker(self, instance: VPSInstance, 
                                terminal_id: str, account_config: dict):
        """Deploy MT5 in Docker container"""
        container = self.docker_client.containers.get(instance.instance_id)
        
        # Create MT5 configuration
        mt5_config = f"""
        [Common]
        Login={account_config.get('login')}
        Server={account_config.get('server')}
        Password={account_config.get('password')}
        AutoTrading=1
        """
        
        # Write config to container
        container.exec_run(f"mkdir -p /opt/mt5/{terminal_id}")
        container.exec_run(f"echo '{mt5_config}' > /opt/mt5/{terminal_id}/config.ini")
        
        # Start MT5 terminal
        container.exec_run(f"wine64 /opt/mt5/terminal64.exe /config:/opt/mt5/{terminal_id}/config.ini", detach=True)
    
    async def _deploy_mt5_process(self, instance: VPSInstance, 
                                 terminal_id: str, account_config: dict):
        """Deploy MT5 as separate process"""
        instance_dir = f"/opt/quantum_vps/{instance.instance_id}"
        terminal_dir = f"{instance_dir}/mt5/{terminal_id}"
        os.makedirs(terminal_dir, exist_ok=True)
        
        # Create terminal configuration
        config_file = f"{terminal_dir}/config.ini"
        with open(config_file, 'w') as f:
            f.write(f"[Common]\n")
            f.write(f"Login={account_config.get('login')}\n")
            f.write(f"Server={account_config.get('server')}\n")
            f.write(f"Password={account_config.get('password')}\n")
            f.write(f"AutoTrading=1\n")
        
        # Start MT5 process (platform specific)
        if platform.system() == 'Windows':
            subprocess.Popen([
                f"{self.config['mt5_config']['terminal_path']}/terminal64.exe",
                f"/config:{config_file}"
            ])
        else:
            # Linux with Wine
            subprocess.Popen([
                "wine64",
                f"{self.config['mt5_config']['terminal_path']}/terminal64.exe",
                f"/config:{config_file}"
            ])
    
    async def monitor_resources(self):
        """Monitor VPS resources and performance"""
        while True:
            for instance_id, instance in self.instances.items():
                if instance.status != 'running':
                    continue
                
                # Get resource usage
                monitor_data = await self._get_resource_usage(instance)
                
                # Log metrics
                logger.info(f"Instance {instance_id}: CPU={monitor_data.cpu_percent}%, "
                          f"Memory={monitor_data.memory_percent}%, "
                          f"Disk={monitor_data.disk_usage}%")
                
                # Check for auto-scaling
                if self.config['auto_scaling']['enabled']:
                    await self._check_auto_scaling(instance, monitor_data)
                
                # Store performance history
                self.performance_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'instance_id': instance_id,
                    'metrics': asdict(monitor_data)
                })
            
            # Clean old history
            self._clean_performance_history()
            
            await asyncio.sleep(self.config['monitoring']['interval_seconds'])
    
    async def _get_resource_usage(self, instance: VPSInstance) -> ResourceMonitor:
        """Get resource usage for instance"""
        if self.docker_client:
            try:
                container = self.docker_client.containers.get(instance.instance_id)
                stats = container.stats(stream=False)
                
                # Calculate CPU usage
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                cpu_percent = (cpu_delta / system_delta) * 100.0
                
                # Calculate memory usage
                memory_usage = stats['memory_stats']['usage']
                memory_limit = stats['memory_stats']['limit']
                memory_percent = (memory_usage / memory_limit) * 100.0
                
                return ResourceMonitor(
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    disk_usage=0.0,  # TODO: Implement disk monitoring
                    network_latency=0.0,  # TODO: Implement network monitoring
                    active_trades=len(instance.mt5_terminals)
                )
            except:
                pass
        
        # Fallback to system monitoring
        return ResourceMonitor(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            network_latency=0.0,
            active_trades=len(instance.mt5_terminals)
        )
    
    async def _check_auto_scaling(self, instance: VPSInstance, 
                                 monitor: ResourceMonitor):
        """Check if auto-scaling is needed"""
        config = self.config['auto_scaling']
        
        # Check if scaling up is needed
        if (monitor.cpu_percent > config['cpu_threshold'] or 
            monitor.memory_percent > config['memory_threshold']):
            
            logger.warning(f"Instance {instance.instance_id} needs scaling: "
                         f"CPU={monitor.cpu_percent}%, Memory={monitor.memory_percent}%")
            
            # Scale up resources
            await self.scale_instance(instance.instance_id, 
                                     cpu_cores=instance.cpu_cores + 1,
                                     ram_mb=instance.ram_mb + 1024)
    
    async def scale_instance(self, instance_id: str, 
                           cpu_cores: int = None, ram_mb: int = None):
        """Scale instance resources"""
        if instance_id not in self.instances:
            raise ValueError(f"Instance {instance_id} not found")
        
        instance = self.instances[instance_id]
        
        if self.docker_client:
            container = self.docker_client.containers.get(instance_id)
            
            # Update container resources
            container.update(
                mem_limit=f"{ram_mb}m" if ram_mb else None,
                cpu_quota=cpu_cores * 100000 if cpu_cores else None
            )
        
        # Update instance configuration
        if cpu_cores:
            instance.cpu_cores = cpu_cores
        if ram_mb:
            instance.ram_mb = ram_mb
        
        logger.info(f"Scaled instance {instance_id}: CPU={instance.cpu_cores}, RAM={instance.ram_mb}MB")
    
    def _clean_performance_history(self):
        """Clean old performance history"""
        retention_hours = self.config['monitoring']['history_retention_hours']
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        self.performance_history = [
            entry for entry in self.performance_history
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]
    
    async def backup_instance(self, instance_id: str, backup_path: str):
        """Backup VPS instance data"""
        if instance_id not in self.instances:
            raise ValueError(f"Instance {instance_id} not found")
        
        instance_dir = f"/opt/quantum_vps/{instance_id}"
        
        # Create backup
        backup_file = f"{backup_path}/{instance_id}_{int(time.time())}.tar.gz"
        
        if platform.system() == 'Windows':
            # Windows backup using PowerShell
            subprocess.run([
                'powershell', '-Command',
                f'Compress-Archive -Path {instance_dir} -DestinationPath {backup_file}'
            ])
        else:
            # Linux backup using tar
            subprocess.run([
                'tar', '-czf', backup_file, instance_dir
            ])
        
        logger.info(f"Backed up instance {instance_id} to {backup_file}")
        return backup_file
    
    async def restore_instance(self, backup_file: str) -> VPSInstance:
        """Restore VPS instance from backup"""
        # Extract instance ID from backup filename
        instance_id = os.path.basename(backup_file).split('_')[0]
        
        # Extract backup
        if platform.system() == 'Windows':
            subprocess.run([
                'powershell', '-Command',
                f'Expand-Archive -Path {backup_file} -DestinationPath /opt/quantum_vps/'
            ])
        else:
            subprocess.run([
                'tar', '-xzf', backup_file, '-C', '/opt/quantum_vps/'
            ])
        
        # Load instance configuration
        config_file = f"/opt/quantum_vps/{instance_id}/config.json"
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Recreate instance
        instance = VPSInstance(
            instance_id=instance_id,
            name=config.get('name', 'restored'),
            cpu_cores=config['resources']['cpu_cores'],
            ram_mb=config['resources']['ram_mb'],
            disk_gb=config['resources']['disk_gb'],
            status='running',
            created_at=datetime.fromisoformat(config['created_at']),
            mt5_terminals=[]
        )
        
        self.instances[instance_id] = instance
        logger.info(f"Restored instance {instance_id} from backup")
        
        return instance
    
    def get_performance_report(self) -> dict:
        """Generate performance report"""
        report = {
            'total_instances': len(self.instances),
            'active_instances': len([i for i in self.instances.values() if i.status == 'running']),
            'total_mt5_terminals': sum(len(i.mt5_terminals) for i in self.instances.values()),
            'resource_usage': {},
            'performance_scores': {}
        }
        
        for instance_id, instance in self.instances.items():
            report['performance_scores'][instance_id] = instance.performance_score
        
        return report

async def main():
    """Main entry point"""
    vps = QuantumVPS()
    
    # Create a VPS instance
    instance = await vps.create_instance(
        name="trading_vps_1",
        cpu_cores=4,
        ram_mb=8192,
        disk_gb=100
    )
    
    # Deploy MT5 terminal
    terminal_id = await vps.deploy_mt5_terminal(
        instance.instance_id,
        {
            'login': '12345678',
            'server': 'ICMarkets-Demo',
            'password': 'demo_password'
        }
    )
    
    # Start monitoring
    await vps.monitor_resources()

if __name__ == "__main__":
    asyncio.run(main())