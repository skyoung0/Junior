import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import matplotlib.patches as patches

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 초기 파라미터 설정
t = np.linspace(0, 2, 1000)  # 시간 축 (2초)
fs = 500  # 샘플링 주파수

# 반송파 파라미터
carrier_freq = 10  # 반송파 주파수 (Hz)
carrier_amp = 1    # 반송파 진폭

# 메시지 신호 파라미터 (아날로그)
message_freq = 2   # 메시지 신호 주파수 (Hz)
message_amp = 0.5  # 메시지 신호 진폭

# 디지털 신호 파라미터
bit_rate = 4       # 비트율 (bps)
digital_bits = [1, 0, 1, 1, 0, 1, 0, 0]  # 디지털 데이터

# 변조 지수
mod_index = 0.5    # AM, FM, PM 변조 지수

# 그래프 초기화
fig = plt.figure(figsize=(14, 10))
plt.subplots_adjust(left=0.3, bottom=0.35, right=0.95, top=0.95)

# 서브플롯 배치
ax1 = plt.subplot(4, 1, 1)  # 메시지/디지털 신호
ax2 = plt.subplot(4, 1, 2)  # 반송파
ax3 = plt.subplot(4, 1, 3)  # 변조된 신호
ax4 = plt.subplot(4, 1, 4)  # 스펙트럼 (FFT)

# 변조 방식 선택
modulation_type = 'AM'

def generate_digital_signal(t, bits, bit_rate):
    """디지털 비트 스트림을 NRZ 신호로 변환"""
    samples_per_bit = int(len(t) / len(bits))
    digital_signal = np.zeros_like(t)
    
    for i, bit in enumerate(bits):
        start_idx = i * samples_per_bit
        end_idx = min((i + 1) * samples_per_bit, len(t))
        digital_signal[start_idx:end_idx] = bit
    
    return digital_signal

def generate_carrier(t, freq, amp):
    """반송파 생성"""
    return amp * np.cos(2 * np.pi * freq * t)

def generate_message(t, freq, amp):
    """아날로그 메시지 신호 생성"""
    return amp * np.cos(2 * np.pi * freq * t)

def modulate_signal(t, carrier_f, carrier_a, message_signal, mod_type, mod_idx=0.5):
    """신호 변조"""
    carrier = generate_carrier(t, carrier_f, carrier_a)
    
    if mod_type == 'AM':  # 진폭 변조
        return (1 + mod_idx * message_signal) * carrier
    
    elif mod_type == 'FM':  # 주파수 변조
        freq_deviation = mod_idx * carrier_f
        phase = 2 * np.pi * carrier_f * t + freq_deviation * np.cumsum(message_signal) * (t[1] - t[0])
        return carrier_a * np.cos(phase)
    
    elif mod_type == 'PM':  # 위상 변조
        phase_deviation = mod_idx * np.pi
        return carrier_a * np.cos(2 * np.pi * carrier_f * t + phase_deviation * message_signal)
    
    elif mod_type == 'ASK':  # 진폭 편이 변조
        return message_signal * carrier
    
    elif mod_type == 'FSK':  # 주파수 편이 변조
        modulated = np.zeros_like(t)
        f1, f2 = carrier_f, carrier_f * 1.5  # 두 개의 주파수 사용
        for i in range(len(t)):
            freq = f1 if message_signal[i] == 0 else f2
            modulated[i] = carrier_a * np.cos(2 * np.pi * freq * t[i])
        return modulated
    
    elif mod_type == 'PSK':  # 위상 편이 변조
        phase_shift = message_signal * np.pi  # 0 또는 π 위상 편이
        return carrier_a * np.cos(2 * np.pi * carrier_f * t + phase_shift)
    
    elif mod_type == 'QAM':  # 구상 진폭 변조 (간단한 4-QAM)
        # 디지털 신호를 I, Q 성분으로 분할
        I_signal = np.zeros_like(t)
        Q_signal = np.zeros_like(t)
        samples_per_symbol = len(t) // (len(digital_bits) // 2)
        
        for i in range(0, len(digital_bits), 2):
            start_idx = (i // 2) * samples_per_symbol
            end_idx = min(((i // 2) + 1) * samples_per_symbol, len(t))
            
            if i < len(digital_bits) - 1:
                I_bit = digital_bits[i]
                Q_bit = digital_bits[i + 1]
                I_signal[start_idx:end_idx] = 1 if I_bit else -1
                Q_signal[start_idx:end_idx] = 1 if Q_bit else -1
        
        I_modulated = I_signal * np.cos(2 * np.pi * carrier_f * t)
        Q_modulated = Q_signal * np.sin(2 * np.pi * carrier_f * t)
        return carrier_a * (I_modulated + Q_modulated)
    
    return carrier

def calculate_fft(signal, fs):
    """FFT 계산"""
    N = len(signal)
    fft_signal = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(N, 1/fs)
    magnitude = np.abs(fft_signal)
    
    # 양의 주파수만 반환
    positive_freq_idx = frequencies >= 0
    return frequencies[positive_freq_idx], magnitude[positive_freq_idx]

def update_plots():
    """모든 플롯 업데이트"""
    global modulation_type
    
    # 메시지 신호 생성
    if modulation_type in ['AM', 'FM', 'PM']:
        message_signal = generate_message(t, message_freq, message_amp)
        signal_label = f'아날로그 메시지 신호 (f={message_freq}Hz)'
    else:
        message_signal = generate_digital_signal(t, digital_bits, bit_rate)
        signal_label = f'디지털 신호 (비트율={bit_rate}bps)'
    
    # 반송파 생성
    carrier = generate_carrier(t, carrier_freq, carrier_amp)
    
    # 변조된 신호 생성
    modulated = modulate_signal(t, carrier_freq, carrier_amp, message_signal, modulation_type, mod_index)
    
    # FFT 계산
    freq_axis, magnitude = calculate_fft(modulated, fs)
    
    # 플롯 클리어
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    
    # 메시지/디지털 신호 플롯
    ax1.plot(t, message_signal, 'b-', linewidth=2)
    ax1.set_ylabel('진폭')
    ax1.set_title(signal_label)
    ax1.grid(True)
    ax1.set_ylim(-1.5, 1.5)
    
    # 반송파 플롯
    ax2.plot(t, carrier, 'r-', linewidth=1)
    ax2.set_ylabel('진폭')
    ax2.set_title(f'반송파 (f={carrier_freq}Hz)')
    ax2.grid(True)
    ax2.set_ylim(-2, 2)
    
    # 변조된 신호 플롯
    ax3.plot(t, modulated, 'g-', linewidth=2)
    ax3.set_ylabel('진폭')
    ax3.set_xlabel('시간 (초)')
    ax3.set_title(f'{modulation_type} 변조 신호')
    ax3.grid(True)
    ax3.set_ylim(-3, 3)
    
    # 주파수 스펙트럼 플롯
    ax4.plot(freq_axis[:len(freq_axis)//2], magnitude[:len(magnitude)//2], 'm-', linewidth=1)
    ax4.set_ylabel('크기')
    ax4.set_xlabel('주파수 (Hz)')
    ax4.set_title('주파수 스펙트럼')
    ax4.grid(True)
    ax4.set_xlim(0, 50)
    
    plt.tight_layout()
    fig.canvas.draw_idle()

# 슬라이더 생성
# 반송파 주파수 슬라이더
ax_carrier_freq = plt.axes([0.3, 0.25, 0.4, 0.03])
slider_carrier_freq = Slider(ax_carrier_freq, '반송파 주파수 (Hz)', 5, 50, valinit=carrier_freq)

# 반송파 진폭 슬라이더
ax_carrier_amp = plt.axes([0.3, 0.21, 0.4, 0.03])
slider_carrier_amp = Slider(ax_carrier_amp, '반송파 진폭', 0.1, 2.0, valinit=carrier_amp)

# 메시지 주파수 슬라이더
ax_message_freq = plt.axes([0.3, 0.17, 0.4, 0.03])
slider_message_freq = Slider(ax_message_freq, '메시지 주파수 (Hz)', 0.5, 10, valinit=message_freq)

# 메시지 진폭 슬라이더
ax_message_amp = plt.axes([0.3, 0.13, 0.4, 0.03])
slider_message_amp = Slider(ax_message_amp, '메시지 진폭', 0.1, 1.0, valinit=message_amp)

# 변조 지수 슬라이더
ax_mod_index = plt.axes([0.3, 0.09, 0.4, 0.03])
slider_mod_index = Slider(ax_mod_index, '변조 지수', 0.1, 2.0, valinit=mod_index)

# 비트율 슬라이더
ax_bit_rate = plt.axes([0.3, 0.05, 0.4, 0.03])
slider_bit_rate = Slider(ax_bit_rate, '비트율 (bps)', 1, 10, valinit=bit_rate, valfmt='%d')

# 변조 방식 선택 라디오 버튼
rax = plt.axes([0.05, 0.4, 0.2, 0.5])
radio = RadioButtons(rax, ['AM', 'FM', 'PM', 'ASK', 'FSK', 'PSK', 'QAM'])

# 범례 추가
legend_ax = plt.axes([0.05, 0.05, 0.2, 0.3])
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)
legend_ax.axis('off')

legend_text = """
변조 방식 설명:
• AM: 진폭 변조
• FM: 주파수 변조  
• PM: 위상 변조
• ASK: 진폭 편이 변조
• FSK: 주파수 편이 변조
• PSK: 위상 편이 변조
• QAM: 구상 진폭 변조
"""
legend_ax.text(0.05, 0.95, legend_text, fontsize=9, verticalalignment='top', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

# 콜백 함수들
def update_carrier_freq(val):
    global carrier_freq
    carrier_freq = val
    update_plots()

def update_carrier_amp(val):
    global carrier_amp
    carrier_amp = val
    update_plots()

def update_message_freq(val):
    global message_freq
    message_freq = val
    update_plots()

def update_message_amp(val):
    global message_amp
    message_amp = val
    update_plots()

def update_mod_index(val):
    global mod_index
    mod_index = val
    update_plots()

def update_bit_rate(val):
    global bit_rate
    bit_rate = int(val)
    update_plots()

def update_modulation(label):
    global modulation_type
    modulation_type = label
    update_plots()

# 이벤트 연결
slider_carrier_freq.on_changed(update_carrier_freq)
slider_carrier_amp.on_changed(update_carrier_amp)
slider_message_freq.on_changed(update_message_freq)
slider_message_amp.on_changed(update_message_amp)
slider_mod_index.on_changed(update_mod_index)
slider_bit_rate.on_changed(update_bit_rate)
radio.on_clicked(update_modulation)

# 초기 플롯
update_plots()

plt.show()