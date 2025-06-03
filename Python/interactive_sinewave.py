import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons

# 초기 파라미터 설정
init_frequency = [1, 2, 3]  # 초기 주파수 (Hz)
init_amplitude = [1, 0.5, 0.3]  # 초기 진폭
init_phase = [0, 0, 0]  # 초기 위상 (라디안)
active = [True, True, True]  # 활성화 상태
t = np.linspace(0, 2, 1000)  # 0부터 2초까지 1000개 샘플

# 변조 관련 파라미터
carrier_freq = 10  # 반송파 주파수
bit_rate = 4       # 디지털 비트율
digital_bits = [1, 0, 1, 1, 0, 1, 0, 0]  # 디지털 데이터
mod_index = 0.5    # 변조 지수

# 그래프 초기화
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(left=0.25, bottom=0.5)

# 변조 방식 (기본값: 사인파 합성)
modulation_mode = '사인파 합성'

# 사인파 계산 함수
def calculate_sine(t, freq, amp, phase, is_active):
    if is_active:
        return amp * np.sin(2 * np.pi * freq * t + phase)
    else:
        return np.zeros_like(t)

# 합성파 계산 함수
def calculate_composite(t, freqs, amps, phases, actives):
    composite = np.zeros_like(t)
    for i in range(3):
        composite += calculate_sine(t, freqs[i], amps[i], phases[i], actives[i])
    return composite

# 디지털 신호 생성 함수
def generate_digital_signal(t, bits, bit_rate):
    samples_per_bit = int(len(t) / len(bits))
    digital_signal = np.zeros_like(t)
    
    for i, bit in enumerate(bits):
        start_idx = i * samples_per_bit
        end_idx = min((i + 1) * samples_per_bit, len(t))
        digital_signal[start_idx:end_idx] = bit
    
    return digital_signal

# 변조 함수들
def modulate_signal(t, carrier_f, message_signal, mod_type, mod_idx=0.5):
    carrier = np.cos(2 * np.pi * carrier_f * t)
    
    if mod_type == 'AM':  # 진폭 변조 - 정확한 공식: s(t) = Ac[1 + m*m(t)]cos(2πfct)
        return (1 + mod_idx * message_signal) * carrier
    
    elif mod_type == 'FM':  # 주파수 변조 - 정확한 공식: s(t) = Ac*cos(2πfct + 2πkf∫m(t)dt)
        # 주파수 편차 = kf * m(t), 적분을 위해 cumulative sum 사용
        freq_deviation = mod_idx * carrier_f  # kf 값
        dt = t[1] - t[0]  # 시간 간격
        # 적분 항: ∫m(t)dt ≈ cumsum(m(t)) * dt
        integral_term = np.cumsum(message_signal) * dt
        instantaneous_phase = 2 * np.pi * carrier_f * t + 2 * np.pi * freq_deviation * integral_term
        return np.cos(instantaneous_phase)
    
    elif mod_type == 'PM':  # 위상 변조 - 정확한 공식: s(t) = Ac*cos(2πfct + kp*m(t))
        phase_deviation = mod_idx * np.pi  # kp 값 (rad)
        return np.cos(2 * np.pi * carrier_f * t + phase_deviation * message_signal)
    
    elif mod_type == 'ASK':  # 진폭 편이 변조 - 정확한 공식: s(t) = A*d(t)*cos(2πfct)
        # d(t)는 디지털 데이터 (0 또는 1)
        # A0 = 0 (0 비트), A1 = 1 (1 비트)
        return message_signal * carrier
    
    elif mod_type == 'FSK':  # 주파수 편이 변조 - 정확한 공식: s(t) = A*cos(2πfit) 
        # f1 for bit 0, f2 for bit 1
        modulated = np.zeros_like(t)
        f1 = carrier_f * 0.8  # 0 비트용 주파수
        f2 = carrier_f * 1.2  # 1 비트용 주파수
        
        for i in range(len(t)):
            freq = f1 if message_signal[i] == 0 else f2
            modulated[i] = np.cos(2 * np.pi * freq * t[i])
        return modulated
    
    elif mod_type == 'PSK':  # 위상 편이 변조 (BPSK) - 정확한 공식: s(t) = A*cos(2πfct + φi)
        # φ0 = 0 for bit 0, φ1 = π for bit 1
        # 즉, bit 0 → +1, bit 1 → -1로 매핑
        bipolar_data = 2 * message_signal - 1  # 0,1 → -1,+1 매핑
        return bipolar_data * carrier
    
    elif mod_type == 'QAM':  # 구상 진폭 변조 (4-QAM) - 정확한 공식: s(t) = I(t)*cos(2πfct) - Q(t)*sin(2πfct)
        I_signal = np.zeros_like(t)
        Q_signal = np.zeros_like(t)
        samples_per_symbol = len(t) // (len(digital_bits) // 2)
        
        # 4-QAM 성좌: (±1, ±1)
        for i in range(0, len(digital_bits)-1, 2):
            start_idx = (i // 2) * samples_per_symbol
            end_idx = min(((i // 2) + 1) * samples_per_symbol, len(t))
            
            I_bit = digital_bits[i]
            Q_bit = digital_bits[i + 1]
            
            # 00→(-1,-1), 01→(-1,+1), 10→(+1,-1), 11→(+1,+1)
            I_signal[start_idx:end_idx] = 1 if I_bit else -1
            Q_signal[start_idx:end_idx] = 1 if Q_bit else -1
        
        # QAM 공식: I*cos(ωt) - Q*sin(ωt)
        I_modulated = I_signal * np.cos(2 * np.pi * carrier_f * t)
        Q_modulated = Q_signal * np.sin(2 * np.pi * carrier_f * t)
        return I_modulated - Q_modulated  # 정확한 QAM 공식 (- 부호)
    
    return carrier

# 초기 사인파 및 합성파 계산
sine_waves = [calculate_sine(t, init_frequency[i], init_amplitude[i], init_phase[i], active[i]) for i in range(3)]
composite_wave = calculate_composite(t, init_frequency, init_amplitude, init_phase, active)

# 개별 사인파 그래프 (상단)
colors = ['red', 'green', 'blue']
sine_lines = []
for i in range(3):
    line, = ax1.plot(t, sine_waves[i], color=colors[i], label=f'사인파 {i + 1}')
    sine_lines.append(line)
ax1.set_xlabel('시간 (초)')
ax1.set_ylabel('진폭')
ax1.set_title('개별 사인파')
ax1.set_ylim(-2, 2)
ax1.legend()
ax1.grid(True)

# 합성파 그래프 (하단)
composite_line, = ax2.plot(t, composite_wave, color='black', linewidth=2)
ax2.set_xlabel('시간 (초)')
ax2.set_ylabel('진폭')
ax2.set_title('합성파')
ax2.set_ylim(-2, 2)
ax2.grid(True)

# 주파수 슬라이더
ax_freq = []
freq_sliders = []
for i in range(3):
    ax_f = plt.axes([0.25, 0.4 - i * 0.05, 0.65, 0.03])
    slider = Slider(
        ax=ax_f,
        label=f'주파수 {i + 1} (Hz)',
        valmin=0.1,
        valmax=10.0,
        valinit=init_frequency[i],
        color=colors[i]
    )
    ax_freq.append(ax_f)
    freq_sliders.append(slider)

# 진폭 슬라이더
ax_amp = []
amp_sliders = []
for i in range(3):
    ax_a = plt.axes([0.25, 0.25 - i * 0.05, 0.65, 0.03])
    slider = Slider(
        ax=ax_a,
        label=f'진폭 {i + 1}',
        valmin=0,
        valmax=2.0,
        valinit=init_amplitude[i],
        color=colors[i]
    )
    ax_amp.append(ax_a)
    amp_sliders.append(slider)

# 위상 슬라이더
ax_phase = []
phase_sliders = []
for i in range(3):
    ax_p = plt.axes([0.25, 0.1 - i * 0.05, 0.65, 0.03])
    slider = Slider(
        ax=ax_p,
        label=f'위상 {i + 1} (rad)',
        valmin=0,
        valmax=2 * np.pi,
        valinit=init_phase[i],
        color=colors[i]
    )
    ax_phase.append(ax_p)
    phase_sliders.append(slider)

# 변조 관련 슬라이더들 (기본적으로 숨김)
# 반송파 주파수 슬라이더
ax_carrier = plt.axes([0.25, -0.05, 0.65, 0.03])
carrier_slider = Slider(
    ax=ax_carrier,
    label='반송파 주파수 (Hz)',
    valmin=5,
    valmax=50,
    valinit=carrier_freq,
    color='purple'
)

# 변조 지수 슬라이더
ax_mod = plt.axes([0.25, -0.1, 0.65, 0.03])
mod_slider = Slider(
    ax=ax_mod,
    label='변조 지수',
    valmin=0.1,
    valmax=2.0,
    valinit=mod_index,
    color='orange'
)

# 활성화/비활성화 체크박스
rax = plt.axes([0.05, 0.7, 0.15, 0.15])
check_labels = [f'사인파 {i + 1}' for i in range(3)]
check = CheckButtons(
    ax=rax,
    labels=check_labels,
    actives=active
)

# 변조 방식 선택 라디오 버튼
radio_ax = plt.axes([0.05, 0.15, 0.15, 0.5])
radio_labels = ['사인파 합성', 'AM', 'FM', 'PM', 'ASK', 'FSK', 'PSK', 'QAM']
radio = RadioButtons(radio_ax, radio_labels)

# 업데이트 함수
def update(val):
    global modulation_mode
    
    # 각 사인파의 파라미터 가져오기
    freqs = [slider.val for slider in freq_sliders]
    amps = [slider.val for slider in amp_sliders]
    phases = [slider.val for slider in phase_sliders]
    
    if modulation_mode == '사인파 합성':
        # 기본 사인파 합성 모드
        # 각 사인파 업데이트
        for i in range(3):
            sine_wave = calculate_sine(t, freqs[i], amps[i], phases[i], active[i])
            sine_lines[i].set_ydata(sine_wave)
        
        # 합성파 업데이트
        composite_wave = calculate_composite(t, freqs, amps, phases, active)
        composite_line.set_ydata(composite_wave)
        
        # 제목 업데이트
        ax1.set_title('개별 사인파')
        ax2.set_title('합성파')
        
    else:
        # 변조 모드
        if modulation_mode in ['AM', 'FM', 'PM']:
            # 아날로그 변조 - 합성파를 메시지 신호로 사용
            message_signal = calculate_composite(t, freqs, amps, phases, active)
            
            # 상단: 메시지 신호(합성파)와 반송파 표시
            carrier_signal = np.cos(2 * np.pi * carrier_slider.val * t)
            sine_lines[0].set_ydata(message_signal)  # 합성된 메시지 신호
            sine_lines[1].set_ydata(carrier_signal)   # 반송파
            sine_lines[2].set_ydata(np.zeros_like(t)) # 빈 공간
            
        else:
            # 디지털 변조 - 디지털 신호 사용 (합성파와 무관)
            message_signal = generate_digital_signal(t, digital_bits, bit_rate)
            
            # 상단: 디지털 신호와 반송파 표시
            carrier_signal = np.cos(2 * np.pi * carrier_slider.val * t)
            sine_lines[0].set_ydata(message_signal)  # 디지털 신호
            sine_lines[1].set_ydata(carrier_signal)   # 반송파
            sine_lines[2].set_ydata(np.zeros_like(t)) # 빈 공간
        
        # 변조된 신호 계산
        modulated_signal = modulate_signal(t, carrier_slider.val, message_signal, modulation_mode, mod_slider.val)
        composite_line.set_ydata(modulated_signal)
        
        # 제목 업데이트
        if modulation_mode in ['AM', 'FM', 'PM']:
            ax1.set_title(f'합성된 메시지 신호 (빨강)와 반송파 (초록) - 아날로그 변조')
        else:
            ax1.set_title(f'디지털 신호 (빨강)와 반송파 (초록) - 디지털 변조')
        ax2.set_title(f'{modulation_mode} 변조된 신호')
    
    fig.canvas.draw_idle()

# 체크박스 콜백 함수
def checkbox_callback(label):
    index = check_labels.index(label)
    active[index] = not active[index]
    update(None)

# 라디오 버튼 콜백 함수
def radio_callback(label):
    global modulation_mode
    modulation_mode = label
    
    # 변조 모드에 따라 슬라이더 가시성 조절
    if label == '사인파 합성':
        ax_carrier.set_visible(False)
        ax_mod.set_visible(False)
    else:
        ax_carrier.set_visible(True)
        ax_mod.set_visible(True)
    
    update(None)

# 이벤트 연결
for slider in freq_sliders + amp_sliders + phase_sliders:
    slider.on_changed(update)
carrier_slider.on_changed(update)
mod_slider.on_changed(update)
check.on_clicked(checkbox_callback)
radio.on_clicked(radio_callback)

# 초기에는 변조 슬라이더 숨기기
ax_carrier.set_visible(False)
ax_mod.set_visible(False)

plt.show()