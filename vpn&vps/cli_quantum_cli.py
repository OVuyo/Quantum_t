#!/usr/bin/env python3
"""
Quantum Trading Command Line Interface
Provides CLI commands for managing the infrastructure
"""

import click
import asyncio
import json
import requests
from typing import Dict, Any
from tabulate import tabulate
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# API endpoint
API_URL = os.getenv("QUANTUM_API_URL", "http://localhost:8000")

@click.group()
@click.option('--api-url', default=API_URL, help='API endpoint URL')
@click.pass_context
def cli(ctx, api_url):
    """Quantum Trading Infrastructure CLI"""
    ctx.ensure_object(dict)
    ctx.obj['API_URL'] = api_url

@cli.group()
@click.pass_context
def vps(ctx):
    """VPS management commands"""
    pass

@vps.command()
@click.option('--name', required=True, help='VPS instance name')
@click.option('--cpu', default=2, help='Number of CPU cores')
@click.option('--ram', default=4096, help='RAM in MB')
@click.option('--disk', default=50, help='Disk space in GB')
@click.pass_context
def create(ctx, name, cpu, ram, disk):
    """Create a new VPS instance"""
    api_url = ctx.obj['API_URL']
    
    data = {
        "name": name,
        "cpu_cores": cpu,
        "ram_mb": ram,
        "disk_gb": disk
    }
    
    response = requests.post(f"{api_url}/vps/create", json=data)
    
    if response.status_code == 200:
        result = response.json()
        click.echo(click.style("✓ VPS instance created successfully!", fg='green'))
        click.echo(f"Instance ID: {result['instance_id']}")
        click.echo(f"Name: {result['details']['name']}")
        click.echo(f"Resources: {cpu} CPU, {ram}MB RAM, {disk}GB Disk")
    else:
        click.echo(click.style(f"✗ Failed to create VPS: {response.text}", fg='red'))

@vps.command()
@click.pass_context
def list(ctx):
    """List all VPS instances"""
    api_url = ctx.obj['API_URL']
    
    response = requests.get(f"{api_url}/vps/list")
    
    if response.status_code == 200:
        instances = response.json()['instances']
        
        if instances:
            headers = ['Instance ID', 'Name', 'Status', 'CPU', 'RAM (MB)', 'MT5 Terminals', 'Created']
            rows = []
            
            for inst in instances:
                rows.append([
                    inst['instance_id'][:20] + '...' if len(inst['instance_id']) > 20 else inst['instance_id'],
                    inst['name'],
                    inst['status'],
                    inst['cpu_cores'],
                    inst['ram_mb'],
                    inst['mt5_terminals'],
                    inst['created_at'][:19]
                ])
            
            click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
        else:
            click.echo("No VPS instances found.")
    else:
        click.echo(click.style(f"✗ Failed to list instances: {response.text}", fg='red'))

@vps.command()
@click.argument('instance_id')
@click.pass_context
def info(ctx, instance_id):
    """Get detailed information about a VPS instance"""
    api_url = ctx.obj['API_URL']
    
    response = requests.get(f"{api_url}/vps/{instance_id}")
    
    if response.status_code == 200:
        data = response.json()
        
        click.echo(click.style(f"\n=== VPS Instance: {data['name']} ===", fg='cyan', bold=True))
        click.echo(f"Instance ID: {data['instance_id']}")
        click.echo(f"Status: {data['status']}")
        
        click.echo(click.style("\nResources:", fg='yellow'))
        click.echo(f"  CPU Cores: {data['resources']['cpu_cores']}")
        click.echo(f"  RAM: {data['resources']['ram_mb']} MB")
        click.echo(f"  Disk: {data['resources']['disk_gb']} GB")
        
        click.echo(click.style("\nCurrent Usage:", fg='yellow'))
        click.echo(f"  CPU: {data['usage']['cpu_percent']:.1f}%")
        click.echo(f"  Memory: {data['usage']['memory_percent']:.1f}%")
        click.echo(f"  Disk: {data['usage']['disk_usage']:.1f}%")
        
        click.echo(click.style("\nMT5 Terminals:", fg='yellow'))
        click.echo(f"  Active: {len(data['mt5_terminals'])}")
        click.echo(f"  Performance Score: {data['performance_score']:.1f}")
    else:
        click.echo(click.style(f"✗ Instance not found: {response.text}", fg='red'))

@vps.command()
@click.argument('instance_id')
@click.option('--cpu', type=int, help='New CPU cores')
@click.option('--ram', type=int, help='New RAM in MB')
@click.pass_context
def scale(ctx, instance_id, cpu, ram):
    """Scale VPS instance resources"""
    api_url = ctx.obj['API_URL']
    
    data = {
        "instance_id": instance_id,
        "cpu_cores": cpu,
        "ram_mb": ram
    }
    
    response = requests.post(f"{api_url}/vps/scale", json=data)
    
    if response.status_code == 200:
        click.echo(click.style("✓ Instance scaled successfully!", fg='green'))
    else:
        click.echo(click.style(f"✗ Failed to scale instance: {response.text}", fg='red'))

@vps.command()
@click.argument('instance_id')
@click.confirmation_option(prompt='Are you sure you want to delete this instance?')
@click.pass_context
def delete(ctx, instance_id):
    """Delete a VPS instance"""
    api_url = ctx.obj['API_URL']
    
    response = requests.delete(f"{api_url}/vps/{instance_id}")
    
    if response.status_code == 200:
        click.echo(click.style("✓ Instance deleted successfully!", fg='green'))
    else:
        click.echo(click.style(f"✗ Failed to delete instance: {response.text}", fg='red'))

@cli.group()
@click.pass_context
def mt5(ctx):
    """MT5 management commands"""
    pass

@mt5.command()
@click.option('--instance', required=True, help='VPS instance ID')
@click.option('--account', required=True, type=click.Path(exists=True), help='Account config JSON file')
@click.pass_context
def deploy(ctx, instance, account):
    """Deploy MT5 terminal on VPS"""
    api_url = ctx.obj['API_URL']
    
    # Load account configuration
    with open(account, 'r') as f:
        account_config = json.load(f)
    
    data = {
        "instance_id": instance,
        "account": account_config
    }
    
    response = requests.post(f"{api_url}/mt5/deploy", json=data)
    
    if response.status_code == 200:
        result = response.json()
        click.echo(click.style("✓ MT5 terminal deployed successfully!", fg='green'))
        click.echo(f"Terminal ID: {result['terminal_id']}")
    else:
        click.echo(click.style(f"✗ Failed to deploy MT5: {response.text}", fg='red'))

@cli.group()
@click.pass_context
def trade(ctx):
    """Trading commands"""
    pass

@trade.command()
@click.option('--instance', required=True, help='VPS instance ID')
@click.option('--symbol', required=True, help='Trading symbol (e.g., EURUSD)')
@click.option('--type', required=True, type=click.Choice(['BUY', 'SELL']), help='Order type')
@click.option('--volume', required=True, type=float, help='Trade volume')
@click.option('--sl', type=float, default=0, help='Stop Loss')
@click.option('--tp', type=float, default=0, help='Take Profit')
@click.option('--comment', default='', help='Trade comment')
@click.pass_context
def execute(ctx, instance, symbol, type, volume, sl, tp, comment):
    """Execute a trade"""
    api_url = ctx.obj['API_URL']
    
    data = {
        "instance_id": instance,
        "symbol": symbol,
        "order_type": type,
        "volume": volume,
        "sl": sl,
        "tp": tp,
        "comment": comment
    }
    
    response = requests.post(f"{api_url}/trade/execute", json=data)
    
    if response.status_code == 200:
        result = response.json()
        if result['status'] == 'success':
            click.echo(click.style("✓ Trade executed successfully!", fg='green'))
            click.echo(f"Ticket: {result['ticket']}")
            click.echo(f"Execution time: {result['execution_time_ms']:.2f}ms")
        else:
            click.echo(click.style(f"✗ Trade failed: {result['message']}", fg='red'))
    else:
        click.echo(click.style(f"✗ Failed to execute trade: {response.text}", fg='red'))

@trade.command()
@click.argument('instance_id')
@click.pass_context
def positions(ctx, instance_id):
    """Show open positions"""
    api_url = ctx.obj['API_URL']
    
    response = requests.get(f"{api_url}/trade/positions/{instance_id}")
    
    if response.status_code == 200:
        positions = response.json()['positions']
        
        if positions:
            headers = ['Ticket', 'Symbol', 'Type', 'Volume', 'Price', 'Profit', 'SL', 'TP']
            rows = []
            
            for ticket, pos in positions.items():
                rows.append([
                    ticket,
                    pos['symbol'],
                    'BUY' if pos['type'] == 0 else 'SELL',
                    pos['volume'],
                    pos['price_open'],
                    f"{pos['profit']:.2f}",
                    pos['sl'],
                    pos['tp']
                ])
            
            click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
        else:
            click.echo("No open positions.")
    else:
        click.echo(click.style(f"✗ Failed to get positions: {response.text}", fg='red'))

@cli.command()
@click.option('--instance', help='VPS instance ID (optional)')
@click.pass_context
def monitor(ctx, instance):
    """Monitor system performance"""
    api_url = ctx.obj['API_URL']
    
    try:
        while True:
            # Clear screen
            click.clear()
            
            # Get metrics
            response = requests.get(f"{api_url}/metrics")
            if response.status_code == 200:
                metrics = response.json()
                
                click.echo(click.style("=== Quantum Trading Monitor ===", fg='cyan', bold=True))
                click.echo(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                click.echo(click.style("\nVPS Status:", fg='yellow'))
                click.echo(f"  Total Instances: {metrics['vps']['total_instances']}")
                click.echo(f"  Active Instances: {metrics['vps']['active_instances']}")
                click.echo(f"  MT5 Terminals: {metrics['vps']['total_mt5_terminals']}")
                
                if metrics['vpn']:
                    click.echo(click.style("\nVPN Performance:", fg='yellow'))
                    click.echo(f"  Total Packets: {metrics['vpn']['total_packets']}")
                    click.echo(f"  Avg Latency: {metrics['vpn']['avg_latency']:.2f}ms")
                    click.echo(f"  Uptime: {metrics['vpn']['uptime_seconds']/3600:.1f} hours")
                
                if metrics['mt5']['active_bridges'] > 0:
                    click.echo(click.style("\nMT5 Trading:", fg='yellow'))
                    click.echo(f"  Total Trades: {metrics['mt5']['total_trades']}")
                    click.echo(f"  Avg Execution: {metrics['mt5']['avg_execution_time']:.2f}ms")
                    click.echo(f"  Active Bridges: {metrics['mt5']['active_bridges']}")
                
                # Get specific instance info if provided
                if instance:
                    response = requests.get(f"{api_url}/vps/{instance}")
                    if response.status_code == 200:
                        data = response.json()
                        click.echo(click.style(f"\nInstance {instance}:", fg='green'))
                        click.echo(f"  CPU Usage: {data['usage']['cpu_percent']:.1f}%")
                        click.echo(f"  Memory Usage: {data['usage']['memory_percent']:.1f}%")
                        click.echo(f"  Performance Score: {data['performance_score']:.1f}")
            
            click.echo("\nPress Ctrl+C to exit...")
            asyncio.run(asyncio.sleep(5))
            
    except KeyboardInterrupt:
        click.echo("\nMonitoring stopped.")

@cli.group()
@click.pass_context
def backup(ctx):
    """Backup and restore commands"""
    pass

@backup.command()
@click.option('--instance', help='VPS instance ID')
@click.option('--all', is_flag=True, help='Backup all instances')
@click.pass_context
def create(ctx, instance, all):
    """Create backup"""
    api_url = ctx.obj['API_URL']
    
    if all:
        # Backup all instances
        response = requests.get(f"{api_url}/vps/list")
        if response.status_code == 200:
            instances = response.json()['instances']
            for inst in instances:
                response = requests.post(f"{api_url}/backup/{inst['instance_id']}")
                if response.status_code == 200:
                    click.echo(f"✓ Backup started for {inst['name']}")
                else:
                    click.echo(f"✗ Failed to backup {inst['name']}")
    elif instance:
        response = requests.post(f"{api_url}/backup/{instance}")
        if response.status_code == 200:
            click.echo(click.style("✓ Backup started successfully!", fg='green'))
        else:
            click.echo(click.style(f"✗ Failed to create backup: {response.text}", fg='red'))
    else:
        click.echo("Please specify --instance or --all")

@backup.command()
@click.option('--file', required=True, help='Backup file path')
@click.pass_context
def restore(ctx, file):
    """Restore from backup"""
    api_url = ctx.obj['API_URL']
    
    data = {"backup_file": file}
    response = requests.post(f"{api_url}/restore", json=data)
    
    if response.status_code == 200:
        result = response.json()
        click.echo(click.style("✓ Instance restored successfully!", fg='green'))
        click.echo(f"Instance ID: {result['instance_id']}")
    else:
        click.echo(click.style(f"✗ Failed to restore: {response.text}", fg='red'))

@cli.command()
@click.pass_context
def status(ctx):
    """Show overall system status"""
    api_url = ctx.obj['API_URL']
    
    # Check API
    try:
        response = requests.get(f"{api_url}/")
        if response.status_code == 200:
            click.echo(click.style("✓ API Server: Online", fg='green'))
        else:
            click.echo(click.style("✗ API Server: Error", fg='red'))
    except:
        click.echo(click.style("✗ API Server: Offline", fg='red'))
        return
    
    # Check VPN
    response = requests.get(f"{api_url}/vpn/status")
    if response.status_code == 200:
        vpn_data = response.json()
        if vpn_data['status'] == 'online':
            click.echo(click.style(f"✓ VPN Server: Online ({vpn_data['active_connections']} connections)", fg='green'))
        else:
            click.echo(click.style("✗ VPN Server: Offline", fg='red'))
    
    # Check VPS
    response = requests.get(f"{api_url}/vps/list")
    if response.status_code == 200:
        instances = response.json()['instances']
        active = len([i for i in instances if i['status'] == 'running'])
        click.echo(click.style(f"✓ VPS Manager: {active}/{len(instances)} instances running", fg='green'))
    
    # Get metrics
    response = requests.get(f"{api_url}/metrics")
    if response.status_code == 200:
        metrics = response.json()
        click.echo(click.style("\nSystem Metrics:", fg='cyan'))
        click.echo(f"  MT5 Terminals: {metrics['vps']['total_mt5_terminals']}")
        click.echo(f"  Total Trades: {metrics['mt5']['total_trades']}")
        if metrics['vpn']:
            click.echo(f"  VPN Latency: {metrics['vpn']['avg_latency']:.2f}ms")

if __name__ == '__main__':
    cli(obj={})