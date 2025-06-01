import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button, TextBox
from matplotlib import font_manager, gridspec
import platform
import os
import sys

# 한글 폰트 설정 (윈도우/맥 환경 모두 지원)
def set_korean_font():
    system = platform.system()
    
    if system == 'Windows':
        font_names = ['Malgun Gothic', 'NanumGothic', 'Gulim']
    elif system == 'Darwin':  # macOS
        font_names = ['AppleGothic', 'NanumGothic', 'STHeiti Light']
    else:  # Linux 등
        font_names = ['NanumGothic', 'UnDotum', 'NotoSansCJK']
    
    # 사용 가능한 폰트 찾기
    available_font = None
    for font in font_names:
        if any(f.name == font for f in font_manager.fontManager.ttflist):
            available_font = font
            break
    
    if available_font:
        plt.rcParams['font.family'] = available_font
    else:
        print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 초기 파라미터 설정
class SignalParameters:
    def __init__(self):
        # 시간 설정
        self.t_start = 0
        self.t_end = 1
        self.t_samples = 1000
        self.t = np.linspace(self.t_start, self.t_end, self.t_samples)
        
        # 기본 신호 파라미터
        self.n_waves = 3
        self.frequencies = [1, 2, 3]  # Hz
        self.amplitudes = [1, 0.5, 0.3]
        self.phases = [0, 0, 0]  # 라디안
        self.active = [True, True, True]
        self.colors = ['red', 'green', 'blue']
        
        # ADC/DAC 파라미터
        self.sample_rate = 20  # Hz
        self.quantize_bits = 4  # 비트
        self.recon_method = '계단형'  # '계단형', '선형', 'Sinc 보간'
        self.auto_range = True  # 자동 범위 조정
        
        # 변조 파라미터
        self.carrier_freq = 10  # Hz
        self.carrier_amp = 1.0
        self.mod_index = 1.0  # 변조 인덱스
        self.mod_type = 'AM'  # 'AM', 'FM', 'PM'
        
        # 아날로그-아날로그 파라미터
        self.filter_cutoff = 5.0  # Hz
        self.filter_q = 1.0  # 품질 계수
        self.filter_type = 'LPF'  # 'LPF', 'HPF', 'BPF', 'BSF', '미분기', '적분기'

# 개선된 신호 처리 함수들
class SignalProcessing:
    @staticmethod
    def calculate_sine(t, freq, amp, phase, is_active):
        """단일 사인파 계산"""
        if is_active:
            return amp * np.sin(2 * np.pi * freq * t + phase)
        else:
            return np.zeros_like(t)
    
    @staticmethod
    def calculate_composite(t, freqs, amps, phases, actives):
        """합성파 생성"""
        composite = np.zeros_like(t)
        for i in range(len(freqs)):
            if i < len(actives) and actives[i]:
                composite += SignalProcessing.calculate_sine(t, freqs[i], amps[i], phases[i], True)
        return composite
    
    @staticmethod
    def sample_signal(signal, t, sample_rate):
        """신호 샘플링"""
        # 샘플링 간격 (초)
        sample_interval = 1.0 / sample_rate
        
        # 샘플링 시간 포인트 계산
        num_samples = int(np.floor((t[-1] - t[0]) * sample_rate)) + 1
        sample_times = np.linspace(t[0], t[0] + (num_samples - 1) * sample_interval, num_samples)
        
        # 샘플링된 값 계산 (원본 신호에서 가장 가까운 시간 값 찾기)
        sampled_values = np.interp(sample_times, t, signal)
        
        return sample_times, sampled_values
    
    @staticmethod
    def quantize_signal(values, bits, auto_range=True, signal_range=None):
        """개선된 신호 양자화"""
        # 양자화 레벨 계산
        levels = 2**bits
        
        if auto_range:
            # 자동 범위: 실제 신호의 최대/최소값 사용
            signal_min = np.min(values)
            signal_max = np.max(values)
            # 범위가 너무 작으면 기본값 사용
            if signal_max - signal_min < 1e-6:
                signal_min, signal_max = -2.0, 2.0
        else:
            # 고정 범위 또는 사용자 정의 범위
            if signal_range is None:
                signal_min, signal_max = -2.0, 2.0
            else:
                signal_min, signal_max = signal_range
        
        # 양자화
        normalized = (values - signal_min) / (signal_max - signal_min)  # 0~1 정규화
        # 범위를 [0, levels-1]로 매핑 후 반올림
        quantized_levels = np.round(normalized * (levels - 1))
        # 범위 제한
        quantized_levels = np.clip(quantized_levels, 0, levels - 1)
        # 다시 원래 신호 범위로 변환
        quantized = (quantized_levels / (levels - 1)) * (signal_max - signal_min) + signal_min
        
        return quantized, (signal_min, signal_max)
        
    @staticmethod
    def reconstruct_signal(sample_times, sample_values, t, method='zero-order'):
        """개선된 신호 재구성"""
        if method == 'zero-order':
            # Zero-order hold (계단 형태)
            reconstructed = np.zeros_like(t)
            
            for i in range(len(t)):
                # t[i]보다 작거나 같은 최대 샘플 시간 인덱스 찾기
                idx = np.where(sample_times <= t[i])[0]
                if len(idx) > 0:
                    reconstructed[i] = sample_values[idx[-1]]
                else:
                    reconstructed[i] = 0
                    
        elif method == 'linear':
            # 선형 보간법
            reconstructed = np.interp(t, sample_times, sample_values)
            
        elif method == 'sinc':
            # 개선된 Sinc 보간법 (벡터화)
            reconstructed = np.zeros_like(t)
            dt = sample_times[1] - sample_times[0]  # 샘플링 간격
            
            # 벡터화된 계산을 위한 메시 그리드
            t_mesh, sample_mesh = np.meshgrid(t, sample_times, indexing='ij')
            time_diff = t_mesh - sample_mesh
            
            # sinc 함수 계산 (0에서의 특이점 처리)
            sinc_values = np.where(np.abs(time_diff) < 1e-10, 
                                 1.0, 
                                 np.sin(np.pi * time_diff / dt) / (np.pi * time_diff / dt))
            
            # 가중합으로 신호 재구성
            reconstructed = np.sum(sinc_values * sample_values, axis=1)
                
        return reconstructed
    
    @staticmethod
    def modulate_signal(t, signal, carrier_freq, carrier_amp, mod_index, mod_type):
        """개선된 신호 변조"""
        # 반송파 신호
        carrier = carrier_amp * np.sin(2 * np.pi * carrier_freq * t)
        
        # 신호 정규화 (안전한 정규화)
        if np.max(np.abs(signal)) > 1e-10:
            normalized_signal = signal / np.max(np.abs(signal))
        else:
            normalized_signal = signal
        
        if mod_type == 'AM':
            # 진폭 변조 (AM): s(t) = A_c [1 + μ*m(t)] * cos(2πf_c*t)
            modulated = carrier_amp * (1 + mod_index * normalized_signal) * np.sin(2 * np.pi * carrier_freq * t)
            
        elif mod_type == 'FM':
            # 주파수 변조 (FM): s(t) = A_c * cos(2πf_c*t + 2πk_f ∫m(τ)dτ)
            # 개선된 적분 계산 (누적 사다리꼴 적분)
            dt = t[1] - t[0]
            integrated_signal = np.cumsum(normalized_signal) * dt
            
            # FM 변조
            modulated = carrier_amp * np.sin(2 * np.pi * carrier_freq * t + mod_index * integrated_signal)
            
        elif mod_type == 'PM':
            # 위상 변조 (PM): s(t) = A_c * cos(2πf_c*t + k_p*m(t))
            modulated = carrier_amp * np.sin(2 * np.pi * carrier_freq * t + mod_index * normalized_signal)
            
        else:
            modulated = carrier  # 기본값
            
        return modulated, carrier
    
    @staticmethod
    def apply_analog_filter(t, signal, cutoff_freq, q_factor, filter_type):
        """개선된 아날로그 필터 적용"""
        # FFT를 이용한 주파수 영역 필터링
        dt = t[1] - t[0]  # 시간 간격
        n = len(signal)   # 신호 길이
        
        # 주파수 영역으로 변환 (FFT)
        fft_signal = np.fft.fft(signal)
        
        # 주파수 배열 생성
        frequencies = np.fft.fftfreq(n, dt)
        
        # 복소 주파수 s = jω 계산
        s = 1j * 2 * np.pi * frequencies
        omega_c = 2 * np.pi * cutoff_freq
        
        # 개선된 필터 전달 함수 계산
        if filter_type == 'LPF':  # 저역 통과 필터 (2차 버터워스)
            transfer_func = (omega_c**2) / (s**2 + np.sqrt(2)*omega_c*s + omega_c**2)
            
        elif filter_type == 'HPF':  # 고역 통과 필터 (2차 버터워스)
            transfer_func = s**2 / (s**2 + np.sqrt(2)*omega_c*s + omega_c**2)
            
        elif filter_type == 'BPF':  # 대역 통과 필터
            transfer_func = (omega_c*s/q_factor) / (s**2 + (omega_c/q_factor)*s + omega_c**2)
            
        elif filter_type == 'BSF':  # 대역 저지 필터
            transfer_func = (s**2 + omega_c**2) / (s**2 + (omega_c/q_factor)*s + omega_c**2)
            
        elif filter_type == '미분기':  # 미분기
            transfer_func = s
            
        elif filter_type == '적분기':  # 적분기
            # 0 주파수에서의 특이점 처리
            transfer_func = np.where(np.abs(s) < 1e-10, 
                                   1e6,  # 매우 큰 값
                                   1.0 / s)
        else:
            # 필터 적용 안함
            return signal
        
        # NaN 및 무한대 값 처리
        transfer_func = np.nan_to_num(transfer_func, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 필터 적용
        filtered_fft = fft_signal * transfer_func
        
        # 역 FFT로 시간 영역으로 변환
        filtered_signal = np.real(np.fft.ifft(filtered_fft))
        
        return filtered_signal
    
    @staticmethod
    def calculate_frequency_response(frequencies, cutoff_freq, q_factor, filter_type):
        """개선된 주파수 응답 계산"""
        # 복소 주파수 s = jω
        s = 1j * 2 * np.pi * frequencies
        omega_c = 2 * np.pi * cutoff_freq
        
        if filter_type == 'LPF':  # 2차 버터워스 LPF
            h = (omega_c**2) / (s**2 + np.sqrt(2)*omega_c*s + omega_c**2)
            
        elif filter_type == 'HPF':  # 2차 버터워스 HPF
            h = s**2 / (s**2 + np.sqrt(2)*omega_c*s + omega_c**2)
            
        elif filter_type == 'BPF':  # 대역 통과 필터
            h = (omega_c*s/q_factor) / (s**2 + (omega_c/q_factor)*s + omega_c**2)
            
        elif filter_type == 'BSF':  # 대역 저지 필터
            h = (s**2 + omega_c**2) / (s**2 + (omega_c/q_factor)*s + omega_c**2)
            
        elif filter_type == '미분기':  # 미분기
            h = s
            
        elif filter_type == '적분기':  # 적분기
            h = np.where(np.abs(s) < 1e-10, 
                        1e6,  # DC에서 매우 큰 값
                        1.0 / s)
        else:
            h = np.ones_like(s)  # 필터 적용 안함
        
        # NaN 및 무한대 값 처리
        h = np.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return np.abs(h)
    
    @staticmethod
    def calculate_snr(original, processed):
        """신호 대 잡음비 계산"""
        noise = original - processed
        signal_power = np.mean(original**2)
        noise_power = np.mean(noise**2)
        
        if noise_power < 1e-10:
            return float('inf')
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        return snr_db
    
    @staticmethod
    def calculate_thd(signal, fundamental_freq, sample_rate):
        """전고조파 왜곡률 계산"""
        # FFT 계산
        fft_signal = np.abs(np.fft.fft(signal))
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
        
        # 기본 주파수 성분 찾기
        fundamental_idx = np.argmin(np.abs(freqs - fundamental_freq))
        fundamental_power = fft_signal[fundamental_idx]**2
        
        # 고조파 성분들 찾기 (2차~5차)
        harmonic_power = 0
        for h in range(2, 6):  # 2차~5차 고조파
            harmonic_freq = h * fundamental_freq
            if harmonic_freq < sample_rate / 2:  # 나이퀴스트 한계 내에서만
                harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonic_power += fft_signal[harmonic_idx]**2
        
        if fundamental_power < 1e-10:
            return 0
        
        thd = np.sqrt(harmonic_power / fundamental_power) * 100
        return thd

class SignalSimulator:
    def __init__(self):
        # 한글 폰트 설정
        set_korean_font()
        
        # 파라미터 초기화
        self.params = SignalParameters()
        
        # ===== 레이아웃 설정 =====
        self.setup_layout()
        
        # ===== 컨트롤 설정 =====
        self.setup_controls()
        
        # ===== 초기 그래프 그리기 =====
        self.update_plot(None)
        
    def setup_layout(self):
        """레이아웃 설정 - 그래프 및 UI 영역 배치"""
        # 메인 그림 생성
        self.fig = plt.figure(figsize=(18, 9))
        self.fig.canvas.manager.set_window_title('개선된 신호 처리 시뮬레이션')
        
        # 3개의 열로 분할 (좌측 컨트롤, 중앙 그래프, 우측 컨트롤)
        self.gs = gridspec.GridSpec(1, 3, width_ratios=[0.7, 4, 1.3])
        
        # 좌측 영역 (변환 선택, 주파수 설정 및 삽입 그래프)
        self.left_panel = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=self.gs[0], height_ratios=[0.7, 1, 1, 1])
        
        # 중앙 영역 (그래프)
        self.center_panel = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=self.gs[1], height_ratios=[1, 1])
        
        # 우측 영역 (상세 설정)
        self.right_panel = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=self.gs[2])
        
        # 각 영역 정의
        # 좌측 패널
        self.mode_panel = plt.subplot(self.left_panel[0])
        self.mode_panel.set_title('변환 선택 부분')
        self.mode_panel.axis('off')
        
        self.freq_panel = plt.subplot(self.left_panel[1])
        self.freq_panel.set_title('주파수 설정 부분')
        self.freq_panel.axis('off')
        
        # 주파수 응답 패널
        self.response_panel = plt.subplot(self.left_panel[2])
        self.response_panel.set_title('주파수 응답')
        self.response_panel.set_visible(False)  # 초기에는 숨김
        
        # 주파수 스펙트럼 패널
        self.spectrum_panel = plt.subplot(self.left_panel[3])
        self.spectrum_panel.set_title('주파수 스펙트럼')
        self.spectrum_panel.set_visible(False)  # 초기에는 숨김
        
        # 중앙 패널
        self.ax1 = plt.subplot(self.center_panel[0])  # 상단 그래프 (원본 신호)
        self.ax1.set_title('주파수 그래프 부분')
        
        self.ax2 = plt.subplot(self.center_panel[1])  # 하단 그래프 (처리된 신호)
        self.ax2.set_title('주파수 그래프 부분')
        
        # 우측 패널
        self.detail_panel = plt.subplot(self.right_panel[0])
        self.detail_panel.set_title('아날로그 to 아날로그 상세')
        self.detail_panel.axis('off')
        
        self.control_panel = plt.subplot(self.right_panel[1])
        self.control_panel.set_title('주파수 슬라이더 부분')
        self.control_panel.axis('off')
        
        # 그래프 설정
        self.ax1.grid(True)
        self.ax2.grid(True)
        self.response_panel.grid(True)
        self.spectrum_panel.grid(True)
        
        # y축 범위 설정
        self.ax1.set_ylim(-2, 2)
        self.ax2.set_ylim(-2, 2)
        
        # 현재 모드
        self.current_mode = 'DAC'
        
    def setup_controls(self):
        """컨트롤 설정 - 슬라이더, 라디오 버튼, 체크박스 등 UI 요소 초기화"""
        # ===== 좌측 패널 컨트롤 =====
        # 변환 선택 부분 (좌측 상단)
        mode_select_ax = plt.axes([0.02, 0.8, 0.15, 0.15])
        self.mode_radio = RadioButtons(
            mode_select_ax,
            ('DAC', 'ADC', '변조', 'A-A'),
            active=0
        )
        self.mode_radio.on_clicked(self.change_mode)
        
        # 주파수 설정 부분 (좌측 하단)
        # 체크박스 설정 (사인파 활성화/비활성화)
        check_ax = plt.axes([0.02, 0.5, 0.15, 0.15])
        check_labels = [f'사인파 {i+1}' for i in range(self.params.n_waves)]
        self.check = CheckButtons(check_ax, check_labels, self.params.active)
        self.check.on_clicked(self.checkbox_callback)
        
        # ===== 우측 패널 슬라이더 =====
        # 초기 파라미터 설정
        init_frequency = self.params.frequencies  # 초기 주파수 (Hz)
        init_amplitude = self.params.amplitudes  # 초기 진폭
        init_phase = self.params.phases  # 초기 위상 (라디안)
        
        # 슬라이더 생성
        self.freq_sliders = []
        self.amp_sliders = []
        self.phase_sliders = []
        
        # 주파수 슬라이더
        for i in range(self.params.n_waves):
            ax = plt.axes([0.82, 0.68 - i * 0.05, 0.12, 0.03])
            slider = Slider(
                ax=ax,
                label=f'주파수 {i+1} (Hz)',
                valmin=0.1,
                valmax=10.0,
                valinit=init_frequency[i],
                color=self.params.colors[i]
            )
            slider.on_changed(self.update_plot)
            self.freq_sliders.append(slider)
        
        # 진폭 슬라이더
        for i in range(self.params.n_waves):
            ax = plt.axes([0.82, 0.53 - i * 0.05, 0.12, 0.03])
            slider = Slider(
                ax=ax,
                label=f'진폭 {i+1}',
                valmin=0,
                valmax=1.5,
                valinit=init_amplitude[i],
                color=self.params.colors[i]
            )
            slider.on_changed(self.update_plot)
            self.amp_sliders.append(slider)
        
        # 위상 슬라이더
        for i in range(self.params.n_waves):
            ax = plt.axes([0.82, 0.38 - i * 0.05, 0.12, 0.03])
            slider = Slider(
                ax=ax,
                label=f'위상 {i+1} (rad)',
                valmin=0,
                valmax=2 * np.pi,
                valinit=init_phase[i],
                color=self.params.colors[i]
            )
            slider.on_changed(self.update_plot)
            self.phase_sliders.append(slider)
        
        # ===== 우측 상단 모드별 컨트롤 =====
        # DAC/ADC 모드 컨트롤
        sample_rate_ax = plt.axes([0.83, 0.9, 0.12, 0.03])
        self.sample_rate_slider = Slider(
            ax=sample_rate_ax,
            label='샘플링 레이트 (Hz)',
            valmin=2,
            valmax=50,
            valinit=self.params.sample_rate
        )
        self.sample_rate_slider.on_changed(self.update_plot)
        
        quantize_bits_ax = plt.axes([0.83, 0.85, 0.12, 0.03])
        self.quantize_bits_slider = Slider(
            ax=quantize_bits_ax,
            label='양자화 비트',
            valmin=1,
            valmax=8,
            valinit=self.params.quantize_bits,
            valstep=1
        )
        self.quantize_bits_slider.on_changed(self.update_plot)
        
        # 변조 관련 컨트롤
        carrier_freq_ax = plt.axes([0.83, 0.9, 0.12, 0.03])
        self.carrier_freq_slider = Slider(
            ax=carrier_freq_ax,
            label='반송파 주파수 (Hz)',
            valmin=5,
            valmax=30,
            valinit=self.params.carrier_freq
        )
        self.carrier_freq_slider.on_changed(self.update_plot)
        carrier_freq_ax.set_visible(False)
        
        mod_index_ax = plt.axes([0.83, 0.85, 0.12, 0.03])
        self.mod_index_slider = Slider(
            ax=mod_index_ax,
            label='변조 지수',
            valmin=0.1,
            valmax=5.0,
            valinit=self.params.mod_index
        )
        self.mod_index_slider.on_changed(self.update_plot)
        mod_index_ax.set_visible(False)
        
        # 아날로그-아날로그 관련 컨트롤
        filter_cutoff_ax = plt.axes([0.83, 0.9, 0.12, 0.03])
        self.filter_cutoff_slider = Slider(
            ax=filter_cutoff_ax,
            label='차단 주파수 (Hz)',
            valmin=0.5,
            valmax=15.0,
            valinit=self.params.filter_cutoff
        )
        self.filter_cutoff_slider.on_changed(self.update_plot)
        filter_cutoff_ax.set_visible(False)
        
        filter_q_ax = plt.axes([0.83, 0.85, 0.12, 0.03])
        self.filter_q_slider = Slider(
            ax=filter_q_ax,
            label='필터 Q',
            valmin=0.1,
            valmax=10.0,
            valinit=self.params.filter_q
        )
        self.filter_q_slider.on_changed(self.update_plot)
        filter_q_ax.set_visible(False)
        
        # ===== 우측 하단 라디오 버튼 (각 모드별) =====
        # DAC 재구성 방법 선택 버튼
        recon_method_ax = plt.axes([0.82, 0.2, 0.14, 0.1])
        self.recon_method_radio = RadioButtons(
            recon_method_ax,
            ('계단형', '선형', 'Sinc 보간'),
            active=0
        )
        self.recon_method_radio.on_clicked(self.update_plot)
        
        # 변조 유형 선택 버튼
        mod_type_ax = plt.axes([0.82, 0.2, 0.14, 0.1])
        self.mod_type_radio = RadioButtons(
            mod_type_ax,
            ('AM', 'FM', 'PM'),
            active=0
        )
        self.mod_type_radio.on_clicked(self.change_mod_type)
        mod_type_ax.set_visible(False)
        
        # 필터 유형 선택 버튼
        filter_type_ax = plt.axes([0.82, 0.05, 0.14, 0.15])
        self.filter_type_radio = RadioButtons(
            filter_type_ax,
            ('LPF', 'HPF', 'BPF', 'BSF', '미분기', '적분기'),
            active=0
        )
        self.filter_type_radio.on_clicked(self.update_plot)
        filter_type_ax.set_visible(False)
            
    def checkbox_callback(self, label):
        """체크박스 콜백 함수"""
        index = [f'사인파 {i+1}' for i in range(self.params.n_waves)].index(label)
        self.params.active[index] = not self.params.active[index]
        self.update_plot(None)
        
    def change_mode(self, label):
        """모드 변경 함수"""
        self.current_mode = label
        
        # 상세 패널 타이틀 업데이트
        self.detail_panel.set_title(f'{label} 상세')
        
        # 주파수 응답 및 스펙트럼 패널 처리
        self.response_panel.set_visible(label == 'A-A')
        self.spectrum_panel.set_visible(label == 'A-A')
        
        # 모든 모드별 컨트롤 숨기기
        # DAC/ADC 컨트롤
        self.sample_rate_slider.ax.set_visible(False)
        self.quantize_bits_slider.ax.set_visible(False)
        self.recon_method_radio.ax.set_visible(False)
        
        # 변조 컨트롤
        self.carrier_freq_slider.ax.set_visible(False)
        self.mod_index_slider.ax.set_visible(False)
        self.mod_type_radio.ax.set_visible(False)
        
        # 아날로그-아날로그 컨트롤
        self.filter_cutoff_slider.ax.set_visible(False)
        self.filter_q_slider.ax.set_visible(False)
        self.filter_type_radio.ax.set_visible(False)
        
        # 모드에 따라 UI 및 그래프 제목 변경
        if label == 'DAC':
            self.ax1.set_title('원본 아날로그 신호')
            self.ax2.set_title('디지털-아날로그 변환 (DAC)')
            
            # DAC 컨트롤 표시
            self.sample_rate_slider.ax.set_visible(True)
            self.quantize_bits_slider.ax.set_visible(True)
            self.recon_method_radio.ax.set_visible(True)
            
        elif label == 'ADC':
            self.ax1.set_title('원본 아날로그 신호')
            self.ax2.set_title('아날로그-디지털 변환 (ADC)')
            
            # ADC 컨트롤 표시
            self.sample_rate_slider.ax.set_visible(True)
            self.quantize_bits_slider.ax.set_visible(True)
            
        elif label == '변조':
            self.ax1.set_title('원본 메시지 신호')
            mod_type = self.params.mod_type
            self.ax2.set_title(f'{mod_type} 변조 신호')
            
            # 변조 컨트롤 표시
            self.carrier_freq_slider.ax.set_visible(True)
            self.mod_index_slider.ax.set_visible(True)
            self.mod_type_radio.ax.set_visible(True)
            
        elif label == 'A-A':
            self.ax1.set_title('원본 아날로그 신호')
            filter_type = self.params.filter_type
            self.ax2.set_title(f'아날로그-아날로그 변환 ({filter_type})')
            
            # 아날로그-아날로그 컨트롤 표시
            self.filter_cutoff_slider.ax.set_visible(True)
            self.filter_q_slider.ax.set_visible(True)
            self.filter_type_radio.ax.set_visible(True)
        
        # 그래프 업데이트
        self.update_plot(None)
        
    def change_mod_type(self, label):
        """변조 유형 변경 함수"""
        self.params.mod_type = label
        self.ax2.set_title(f'{label} 변조 신호')
        self.update_plot(None)
    
    # ===== 그래프 업데이트 함수 =====
    def update_plot(self, val):
        """그래프 업데이트 함수"""
        # 슬라이더 값 읽기
        self.params.frequencies = [slider.val for slider in self.freq_sliders]
        self.params.amplitudes = [slider.val for slider in self.amp_sliders]
        self.params.phases = [slider.val for slider in self.phase_sliders]
        
        # 모드별 파라미터 업데이트
        if self.current_mode in ['DAC', 'ADC']:
            self.params.sample_rate = self.sample_rate_slider.val
            self.params.quantize_bits = int(self.quantize_bits_slider.val)
            
        if self.current_mode == 'DAC':
            self.params.recon_method = self.recon_method_radio.value_selected
            
        elif self.current_mode == '변조':
            self.params.carrier_freq = self.carrier_freq_slider.val
            self.params.mod_index = self.mod_index_slider.val
            self.params.mod_type = self.mod_type_radio.value_selected
            
        elif self.current_mode == 'A-A':
            self.params.filter_cutoff = self.filter_cutoff_slider.val
            self.params.filter_q = self.filter_q_slider.val
            self.params.filter_type = self.filter_type_radio.value_selected
        
        # 모든 그래프 지우기
        self.ax1.clear()
        self.ax2.clear()
        
        # 그리드 다시 설정
        self.ax1.grid(True)
        self.ax2.grid(True)
        
        # y축 범위 다시 설정
        self.ax1.set_ylim(-2, 2)
        self.ax2.set_ylim(-2, 2)
        
        # 원본 신호 계산
        original_waves = []
        for i in range(self.params.n_waves):
            wave = SignalProcessing.calculate_sine(
                self.params.t,
                self.params.frequencies[i],
                self.params.amplitudes[i],
                self.params.phases[i],
                self.params.active[i]
            )
            original_waves.append(wave)
            
        # 합성 신호 계산
        composite_wave = SignalProcessing.calculate_composite(
            self.params.t,
            self.params.frequencies,
            self.params.amplitudes,
            self.params.phases,
            self.params.active
        )
        
        # 원본 개별 파형 그리기
        for i in range(self.params.n_waves):
            if self.params.active[i]:
                self.ax1.plot(
                    self.params.t,
                    original_waves[i],
                    color=self.params.colors[i],
                    alpha=0.5,
                    label=f'사인파 {i+1} ({self.params.frequencies[i]:.1f}Hz)'
                )
        
        # 원본 합성 파형 그리기
        self.ax1.plot(self.params.t, composite_wave, 'k-', label='합성파')
        self.ax1.set_xlabel('시간 (초)')
        self.ax1.set_ylabel('진폭')
        self.ax1.legend(loc='upper right')
        
        # 모드에 따라 다른 처리 적용
        if self.current_mode == 'DAC':
            # DAC 과정 시뮬레이션
            sample_times, sampled_values = SignalProcessing.sample_signal(
                composite_wave,
                self.params.t,
                self.params.sample_rate
            )
            
            # 개선된 양자화
            quantized_values, signal_range = SignalProcessing.quantize_signal(
                sampled_values,
                self.params.quantize_bits,
                auto_range=self.params.auto_range
            )
            
            # 재구성 방법 선택
            recon_methods = {'계단형': 'zero-order', '선형': 'linear', 'Sinc 보간': 'sinc'}
            selected_method = self.params.recon_method
            
            # 신호 재구성
            reconstructed = SignalProcessing.reconstruct_signal(
                sample_times,
                quantized_values,
                self.params.t,
                recon_methods[selected_method]
            )
            
            # 샘플링 포인트 그리기
            self.ax2.stem(
                sample_times,
                quantized_values,
                'r',
                markerfmt='ro',
                linefmt='r-',
                basefmt='k-',
                label='샘플링 & 양자화'
            )
            
            # 재구성 신호 그리기
            self.ax2.plot(
                self.params.t,
                reconstructed,
                'b-',
                label=f'재구성 ({selected_method})'
            )
            
            # 원본 신호 (참조용)
            self.ax2.plot(
                self.params.t,
                composite_wave,
                'k--',
                alpha=0.5,
                label='원본'
            )
            
            # 양자화 레벨 표시
            levels = 2**self.params.quantize_bits
            level_values = np.linspace(signal_range[0], signal_range[1], levels)
            for level in level_values:
                self.ax2.axhline(y=level, color='g', linestyle='-', alpha=0.2)
            
            # 나이퀴스트 주파수 표시
            nyquist = self.params.sample_rate / 2
            max_freq = max([f for f, a in zip(self.params.frequencies, self.params.active) if a]) if any(self.params.active) else 0
            if max_freq > nyquist:
                self.ax2.text(
                    0.02,
                    0.98,
                    f'경고: 최대 주파수({max_freq:.1f}Hz)가 나이퀴스트 주파수({nyquist:.1f}Hz)를 초과함',
                    transform=self.ax2.transAxes,
                    color='red',
                    verticalalignment='top'
                )
            
            # SNR 계산 및 표시
            snr = SignalProcessing.calculate_snr(composite_wave, reconstructed)
            if snr != float('inf'):
                self.ax2.text(
                    0.02,
                    0.88,
                    f'SNR: {snr:.1f} dB',
                    transform=self.ax2.transAxes,
                    color='blue',
                    verticalalignment='top'
                )
            
            self.ax2.set_xlabel('시간 (초)')
            self.ax2.set_ylabel('진폭')
            self.ax2.legend(loc='upper right')
            
        elif self.current_mode == 'ADC':
            # ADC 과정 시뮬레이션
            # 샘플링
            sample_times, sampled_values = SignalProcessing.sample_signal(
                composite_wave,
                self.params.t,
                self.params.sample_rate
            )
            
            # 개선된 양자화
            quantized_values, signal_range = SignalProcessing.quantize_signal(
                sampled_values,
                self.params.quantize_bits,
                auto_range=self.params.auto_range
            )
            
            # 양자화 오차 계산
            quantization_error = sampled_values - quantized_values
            
            # 개선된 디지털 표현
            binary_values = []
            levels = 2**self.params.quantize_bits
            for value in quantized_values:
                # 양자화 레벨 인덱스 계산
                normalized = (value - signal_range[0]) / (signal_range[1] - signal_range[0])
                level_idx = int(np.round(normalized * (levels - 1)))
                level_idx = np.clip(level_idx, 0, levels - 1)
                
                # 이진 표현
                binary = format(level_idx, f'0{self.params.quantize_bits}b')
                binary_values.append(binary)
            
            # 샘플링 포인트와 양자화 레벨 그리기
            self.ax2.stem(
                sample_times,
                quantized_values,
                'r',
                markerfmt='ro',
                linefmt='r-',
                basefmt='k-',
                label='양자화된 샘플'
            )
            
            # 양자화 레벨 표시
            level_values = np.linspace(signal_range[0], signal_range[1], levels)
            for level in level_values:
                self.ax2.axhline(y=level, color='g', linestyle='-', alpha=0.2)
            
            # 이진 값 표시
            for i, (t, v, b) in enumerate(zip(sample_times, quantized_values, binary_values)):
                if i % 2 == 0:  # 가독성을 위해 일부만 표시
                    self.ax2.text(t, v + 0.1, b, ha='center', va='bottom', fontsize=8)
            
            # 양자화 오차 표시
            self.ax2.plot(
                sample_times,
                quantization_error,
                'm-',
                alpha=0.5,
                label='양자화 오차'
            )
            
            # 나이퀴스트 주파수 표시
            nyquist = self.params.sample_rate / 2
            max_freq = max([f for f, a in zip(self.params.frequencies, self.params.active) if a]) if any(self.params.active) else 0
            if max_freq > nyquist:
                self.ax2.text(
                    0.02,
                    0.98,
                    f'경고: 최대 주파수({max_freq:.1f}Hz)가 나이퀴스트 주파수({nyquist:.1f}Hz)를 초과함',
                    transform=self.ax2.transAxes,
                    color='red',
                    verticalalignment='top'
                )
            
            # 양자화 노이즈 전력 계산
            quantization_noise_power = np.mean(quantization_error**2)
            self.ax2.text(
                0.02,
                0.88,
                f'양자화 노이즈: {quantization_noise_power:.4f}',
                transform=self.ax2.transAxes,
                color='magenta',
                verticalalignment='top'
            )
            
            # 원본 신호 (참조용)
            self.ax2.plot(
                self.params.t,
                composite_wave,
                'k--',
                alpha=0.3,
                label='원본'
            )
            
            self.ax2.set_xlabel('시간 (초)')
            self.ax2.set_ylabel('진폭')
            self.ax2.legend(loc='upper right')
            
        elif self.current_mode == '변조':
            # 변조 과정 시뮬레이션
            modulated, carrier = SignalProcessing.modulate_signal(
                self.params.t,
                composite_wave,
                self.params.carrier_freq,
                self.params.carrier_amp,
                self.params.mod_index,
                self.params.mod_type
            )
            
            # 반송파 그리기
            self.ax2.plot(
                self.params.t,
                carrier,
                'g-',
                alpha=0.5,
                label=f'반송파 ({self.params.carrier_freq:.1f}Hz)'
            )
            
            # 변조 신호 그리기
            self.ax2.plot(
                self.params.t,
                modulated,
                'b-',
                label=f'{self.params.mod_type} 변조 신호'
            )
            
            # 메시지 신호 (참조용)
            self.ax2.plot(
                self.params.t,
                composite_wave,
                'k--',
                alpha=0.5,
                label='메시지 신호'
            )
            
            # 변조 유형에 따른 추가 정보 표시
            if self.params.mod_type == 'AM':
                self.ax2.text(
                    0.02,
                    0.98,
                    f'변조 지수(μ): {self.params.mod_index:.2f}',
                    transform=self.ax2.transAxes,
                    verticalalignment='top'
                )
                # AM 대역폭 = 2 * 메시지 최대 주파수
                max_freq = max([f for f, a in zip(self.params.frequencies, self.params.active) if a]) if any(self.params.active) else 0
                self.ax2.text(
                    0.02,
                    0.93,
                    f'대역폭: {2 * max_freq:.1f}Hz',
                    transform=self.ax2.transAxes,
                    verticalalignment='top'
                )
                
                # 변조 효율성 계산
                if self.params.mod_index <= 1:
                    efficiency = (self.params.mod_index**2) / (2 + self.params.mod_index**2) * 100
                    self.ax2.text(
                        0.02,
                        0.88,
                        f'변조 효율: {efficiency:.1f}%',
                        transform=self.ax2.transAxes,
                        verticalalignment='top'
                    )
                    
            elif self.params.mod_type == 'FM':
                # FM 주파수 편이 = 변조 지수 * 최대 주파수
                max_freq = max([f for f, a in zip(self.params.frequencies, self.params.active) if a]) if any(self.params.active) else 0
                freq_deviation = self.params.mod_index * max_freq
                self.ax2.text(
                    0.02,
                    0.98,
                    f'주파수 편이: {freq_deviation:.2f}Hz',
                    transform=self.ax2.transAxes,
                    verticalalignment='top'
                )
                # FM 대역폭 (카슨의 법칙) ≈ 2(Δf + fm)
                carson_bandwidth = 2 * (freq_deviation + max_freq)
                self.ax2.text(
                    0.02,
                    0.93,
                    f'대역폭(카슨): {carson_bandwidth:.1f}Hz',
                    transform=self.ax2.transAxes,
                    verticalalignment='top'
                )
                
                # 변조 지수 β
                if max_freq > 0:
                    beta = freq_deviation / max_freq
                    self.ax2.text(
                        0.02,
                        0.88,
                        f'변조 지수 β: {beta:.2f}',
                        transform=self.ax2.transAxes,
                        verticalalignment='top'
                    )
                    
            elif self.params.mod_type == 'PM':
                self.ax2.text(
                    0.02,
                    0.98,
                    f'위상 편이: {self.params.mod_index:.2f}rad',
                    transform=self.ax2.transAxes,
                    verticalalignment='top'
                )
                
                # 위상 편이를 도 단위로도 표시
                phase_deviation_deg = self.params.mod_index * 180 / np.pi
                self.ax2.text(
                    0.02,
                    0.93,
                    f'위상 편이: {phase_deviation_deg:.1f}°',
                    transform=self.ax2.transAxes,
                    verticalalignment='top'
                )
            
            self.ax2.set_xlabel('시간 (초)')
            self.ax2.set_ylabel('진폭')
            self.ax2.legend(loc='upper right')
            
        elif self.current_mode == 'A-A':
            # 아날로그-아날로그 시뮬레이션
            
            # 개선된 필터 적용
            filtered_signal = SignalProcessing.apply_analog_filter(
                self.params.t,
                composite_wave,
                self.params.filter_cutoff,
                self.params.filter_q,
                self.params.filter_type
            )
            
            # 필터링된 신호 그리기
            self.ax2.plot(
                self.params.t,
                filtered_signal,
                'b-',
                label=f'{self.params.filter_type} 필터링'
            )
            
            # 원본 신호 (참조용)
            self.ax2.plot(
                self.params.t,
                composite_wave,
                'k--',
                alpha=0.5,
                label='원본'
            )
            
            # 필터 정보 표시
            if self.params.filter_type in ['LPF', 'HPF']:
                self.ax2.text(
                    0.02,
                    0.98,
                    f'차단 주파수: {self.params.filter_cutoff:.1f}Hz',
                    transform=self.ax2.transAxes,
                    verticalalignment='top'
                )
            elif self.params.filter_type in ['BPF', 'BSF']:
                self.ax2.text(
                    0.02,
                    0.98,
                    f'중심 주파수: {self.params.filter_cutoff:.1f}Hz, Q: {self.params.filter_q:.1f}',
                    transform=self.ax2.transAxes,
                    verticalalignment='top'
                )
                
                # 대역폭 계산
                bandwidth = self.params.filter_cutoff / self.params.filter_q
                self.ax2.text(
                    0.02,
                    0.93,
                    f'대역폭: {bandwidth:.2f}Hz',
                    transform=self.ax2.transAxes,
                    verticalalignment='top'
                )
            
            # 개선된 주파수 응답 및 스펙트럼 계산
            dt = self.params.t[1] - self.params.t[0]  # 시간 간격
            n = len(composite_wave)   # 신호 길이
            
            # FFT 계산
            original_fft = np.abs(np.fft.fft(composite_wave))
            filtered_fft = np.abs(np.fft.fft(filtered_signal))
            
            # 주파수 배열 계산
            freq = np.fft.fftfreq(n, dt)
            
            # 양의 주파수만 표시 (절반만)
            positive_freq_idx = freq > 0
            
            # 개선된 주파수 응답 계산
            freq_range = np.logspace(-1, 1.5, 100)  # 0.1Hz ~ 31.6Hz, 로그 스케일
            response = SignalProcessing.calculate_frequency_response(
                freq_range,
                self.params.filter_cutoff,
                self.params.filter_q,
                self.params.filter_type
            )
            
            # 주파수 응답 패널 그리기
            self.response_panel.clear()
            self.response_panel.semilogx(freq_range, 20*np.log10(response + 1e-10))  # dB 스케일
            self.response_panel.set_xlabel('주파수 (Hz)')
            self.response_panel.set_ylabel('크기 (dB)')
            self.response_panel.set_title('주파수 응답 (dB)')
            self.response_panel.grid(True, which="both", ls="-", alpha=0.3)
            self.response_panel.set_visible(True)
            
            # 차단 주파수에서 -3dB 선 표시
            if self.params.filter_type in ['LPF', 'HPF', 'BPF', 'BSF']:
                self.response_panel.axvline(x=self.params.filter_cutoff, color='r', linestyle='--', alpha=0.7, label='-3dB 점')
                self.response_panel.axhline(y=-3, color='r', linestyle='--', alpha=0.7)
                self.response_panel.legend(fontsize='small')
            
            # 주파수 스펙트럼 패널 그리기
            self.spectrum_panel.clear()
            max_freq_display = min(10, self.params.sample_rate/2) if hasattr(self.params, 'sample_rate') else 10
            freq_mask = (freq > 0) & (freq <= max_freq_display)
            
            self.spectrum_panel.plot(freq[freq_mask], original_fft[freq_mask], 'k--', alpha=0.5, label='원본')
            self.spectrum_panel.plot(freq[freq_mask], filtered_fft[freq_mask], 'b-', label='필터링')
            self.spectrum_panel.set_xlabel('주파수 (Hz)')
            self.spectrum_panel.set_ylabel('크기')
            self.spectrum_panel.set_title('주파수 스펙트럼')
            self.spectrum_panel.set_xlim(0, max_freq_display)
            self.spectrum_panel.legend(fontsize='small')
            self.spectrum_panel.grid(True)
            self.spectrum_panel.set_visible(True)
            
            # SNR 계산
            snr = SignalProcessing.calculate_snr(composite_wave, filtered_signal)
            if snr != float('inf'):
                self.ax2.text(
                    0.02,
                    0.88,
                    f'SNR: {snr:.1f} dB',
                    transform=self.ax2.transAxes,
                    verticalalignment='top'
                )
            
            self.ax2.set_xlabel('시간 (초)')
            self.ax2.set_ylabel('진폭')
            self.ax2.set_title(f'아날로그-아날로그 변환 ({self.params.filter_type})')
            self.ax2.legend(loc='upper right')
        
        # 그래프 다시 그리기
        self.fig.canvas.draw_idle()

# 메인 실행
if __name__ == "__main__":
    simulator = SignalSimulator()
    plt.show()