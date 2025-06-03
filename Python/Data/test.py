import torch
import sys
import platform

print("=== 시스템 정보 ===")
print(f"Python 버전: {sys.version}")
print(f"플랫폼: {platform.platform()}")
print(f"아키텍처: {platform.architecture()}")

print("\n=== PyTorch 정보 ===")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"CUDA 버전 (PyTorch): {torch.version.cuda}")
print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
print(f"cuDNN 활성화: {torch.backends.cudnn.enabled}")

if torch.cuda.is_available():
    print("\n=== GPU 상세 정보 ===")
    print(f"GPU 개수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  - 메모리: {props.total_memory / 1024**3:.1f} GB")
        print(f"  - 컴퓨트 능력: {props.major}.{props.minor}")
        print(f"  - 멀티프로세서: {props.multi_processor_count}")
    
    print(f"\n현재 GPU: {torch.cuda.current_device()}")
    print(f"GPU 이름: {torch.cuda.get_device_name()}")
    
    print("\n=== GPU 메모리 테스트 ===")
    print(f"사용 가능 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"현재 할당된 메모리: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
    print(f"캐시된 메모리: {torch.cuda.memory_reserved() / 1024**3:.3f} GB")
    
    print("\n=== GPU 연산 테스트 ===")
    try:
        # 큰 텐서로 테스트
        device = torch.device('cuda:0')
        print(f"디바이스: {device}")
        
        # GPU 메모리 사용량 체크
        torch.cuda.empty_cache()
        
        x = torch.randn(5000, 5000, device=device)
        y = torch.randn(5000, 5000, device=device)
        
        print(f"텐서 생성 후 메모리: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
        
        # 행렬 곱셈
        import time
        start_time = time.time()
        z = torch.mm(x, y)
        end_time = time.time()
        
        print(f"✅ 행렬 곱셈 성공!")
        print(f"   - 연산 시간: {end_time - start_time:.3f}초")
        print(f"   - 결과 크기: {z.shape}")
        print(f"   - 결과 디바이스: {z.device}")
        print(f"연산 후 메모리: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
        
        # 정리
        del x, y, z
        torch.cuda.empty_cache()
        print("GPU 메모리 정리 완료")
        
    except Exception as e:
        print(f"❌ GPU 연산 실패: {e}")
        
else:
    print("❌ CUDA 사용 불가능")
    print("\n가능한 원인:")
    print("- NVIDIA 드라이버 문제")
    print("- CUDA 설치 문제") 
    print("- PyTorch CUDA 버전 불일치")
    print("- 환경 변수 문제")

print("\n🎉 완전한 GPU 테스트 완료!")