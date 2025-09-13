"""
Performance benchmarks for Quantum Trading Infrastructure
"""

import asyncio
import time
import statistics
from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.vps.vps_manager import QuantumVPS
from infrastructure.vpn.vpn_server import QuantumVPN
from infrastructure.integration.mt5_bridge import MT5Bridge, TradingSignal, OrderType

class PerformanceBenchmark:
    """Performance benchmarking suite"""
    
    def __init__(self):
        self.results = {}
    
    async def benchmark_vps_creation(self, iterations: int = 10) -> dict:
        """Benchmark VPS instance creation"""
        vps = QuantumVPS()
        times = []
        
        for i in range(iterations):
            start = time.perf_counter()
            instance = await vps.create_instance(f"benchmark_{i}")
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        return {
            'min': min(times),
            'max': max(times),
            'avg': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def benchmark_route_finding(self, iterations: int = 100) -> dict:
        """Benchmark VPN route finding"""
        vpn = QuantumVPN()
        brokers = ['ICMarkets', 'Pepperstone', 'FTMO', 'XM']
        times = []
        
        for _ in range(iterations):
            for broker in brokers:
                start = time.perf_counter()
                route = vpn.find_best_route(broker)
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        return {
            'min': min(times),
            'max': max(times),
            'avg': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def benchmark_latency_measurement(self, iterations: int = 50) -> dict:
        """Benchmark network latency measurement"""
        vpn = QuantumVPN()
        test_ips = ['8.8.8.8', '1.1.1.1', '103.86.98.0']
        times = []
        latencies = []
        
        for _ in range(iterations):
            for ip in test_ips:
                start = time.perf_counter()
                latency = vpn.measure_latency(ip)
                end = time.perf_counter()
                times.append((end - start) * 1000)
                latencies.append(latency)
        
        return {
            'measurement_time': {
                'min': min(times),
                'max': max(times),
                'avg': statistics.mean(times)
            },
            'latency': {
                'min': min(latencies),
                'max': max(latencies),
                'avg': statistics.mean(latencies)
            }
        }
    
    async def benchmark_trade_execution(self, iterations: int = 20) -> dict:
        """Benchmark simulated trade execution"""
        times = []
        
        for i in range(iterations):
            signal = TradingSignal(
                symbol='EURUSD',
                order_type=OrderType.BUY,
                volume=0.01,
                price=1.0850,
                sl=1.0800,
                tp=1.0900,
                comment=f'Benchmark {i}',
                magic=12345
            )
            
            # Simulate trade execution time
            start = time.perf_counter()
            await asyncio.sleep(0.001)  # Simulate 1ms execution
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return {
            'min': min(times),
            'max': max(times),
            'avg': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0
        }
    
    async def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("=" * 60)
        print("QUANTUM TRADING INFRASTRUCTURE PERFORMANCE BENCHMARKS")
        print("=" * 60)
        
        # VPS Creation Benchmark
        print("\n1. VPS Instance Creation Benchmark")
        print("-" * 40)
        vps_results = await self.benchmark_vps_creation(5)
        for key, value in vps_results.items():
            print(f"  {key.capitalize()}: {value:.2f} ms")
        self.results['vps_creation'] = vps_results
        
        # Route Finding Benchmark
        print("\n2. VPN Route Finding Benchmark")
        print("-" * 40)
        route_results = self.benchmark_route_finding(50)
        for key, value in route_results.items():
            print(f"  {key.capitalize()}: {value:.4f} ms")
        self.results['route_finding'] = route_results
        
        # Latency Measurement Benchmark
        print("\n3. Network Latency Measurement Benchmark")
        print("-" * 40)
        latency_results = self.benchmark_latency_measurement(20)
        print("  Measurement Time:")
        for key, value in latency_results['measurement_time'].items():
            print(f"    {key.capitalize()}: {value:.2f} ms")
        print("  Network Latency:")
        for key, value in latency_results['latency'].items():
            print(f"    {key.capitalize()}: {value:.2f} ms")
        self.results['latency'] = latency_results
        
        # Trade Execution Benchmark
        print("\n4. Trade Execution Simulation Benchmark")
        print("-" * 40)
        trade_results = await self.benchmark_trade_execution(50)
        for key, value in trade_results.items():
            print(f"  {key.capitalize()}: {value:.4f} ms")
        self.results['trade_execution'] = trade_results
        
        # Summary
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"VPS Creation Avg: {vps_results['avg']:.2f} ms")
        print(f"Route Finding Avg: {route_results['avg']:.4f} ms")
        print(f"Network Latency Avg: {latency_results['latency']['avg']:.2f} ms")
        print(f"Trade Execution Avg: {trade_results['avg']:.4f} ms")
        
        # Performance Grade
        total_score = 100
        if vps_results['avg'] > 1000:
            total_score -= 20
        if route_results['avg'] > 1:
            total_score -= 10
        if latency_results['latency']['avg'] > 50:
            total_score -= 20
        if trade_results['avg'] > 5:
            total_score -= 10
        
        grade = 'A+' if total_score >= 95 else \
                'A' if total_score >= 90 else \
                'B' if total_score >= 80 else \
                'C' if total_score >= 70 else 'D'
        
        print(f"\nPerformance Grade: {grade} ({total_score}/100)")
        
        return self.results

async def main():
    """Run performance benchmarks"""
    benchmark = PerformanceBenchmark()
    results = await benchmark.run_all_benchmarks()
    
    # Save results to file
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to benchmark_results.json")

if __name__ == "__main__":
    asyncio.run(main())