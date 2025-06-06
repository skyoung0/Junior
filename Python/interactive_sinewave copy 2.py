import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
import matplotlib.patches as patches
import platform
import matplotlib.font_manager as fm

# 크로스 플랫폼 한글 폰트 설정
def setup_korean_font():
    """운영체제별 한글 폰트 설정"""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        korean_fonts = ['AppleGothic', 'Arial Unicode MS', 'AppleSDGothicNeo-Regular']
    elif system == 'Windows':  # Windows
        korean_fonts = ['Malgun Gothic', 'Microsoft YaHei', 'SimHei']
    else:  # Linux 및 기타
        korean_fonts = ['DejaVu Sans', 'Liberation Sans', 'Arial']
    
    # 사용 가능한 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in korean_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            break
    else:
        # 한글 폰트가 없는 경우 기본 폰트 사용하고 경고
        print("Warning: 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 설정 적용
setup_korean_font()

# 초기 파라미터 설정
init_frequency = [1, 2, 3]  # 초기 주파수 (Hz)
init_amplitude = [1, 0.5, 0.3]  # 초기 진폭
init_phase = [0, 0, 0]  # 초기 위상 (라디안)
active = [True, True, True]  # 활성화 상태
t = np.linspace(0, 2, 1000)  # 0부터 2초까지 1000개 샘플

# 변조 파라미터
carrier_freq = 10  # 반송파 주파수
modulation_index = 0.5  # 변조 지수
digital_data = np.array([1, 0, 1, 1, 0, 1, 0, 0])  # 디지털 데이터
current_modulation = 'None'  # 현재 변조 방식

# 디지털 변조 세부 파라미터
bit_duration = 0.25  # 비트 지속 시간
ask_levels = [0.2, 1.0]  # ASK 레벨 (0비트, 1비트)
fsk_deviation = 3  # FSK 주파수 편차
psk_type = 'BPSK'  # PSK 타입 (BPSK, QPSK)
qam_type = '4-QAM'  # QAM 타입 (4-QAM, 16-QAM)
digital_input_mode = 'preset'  # 입력 모드 (preset, manual, random)

# 그래프 초기화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(left=0.25, bottom=0.55, right=0.95, top=0.95)

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
def generate_digital_signal(t, data, bit_duration=0.25):
    signal = np.zeros_like(t)
    for i, bit in enumerate(data):
        start_time = i * bit_duration
        end_time = (i + 1) * bit_duration
        mask = (t >= start_time) & (t < end_time)
        signal[mask] = bit
    return signal

# 변조 함수들
def amplitude_modulation(t, message, carrier_freq, mod_index):
    """진폭 변조 (AM) - 완전한 구현"""
    # 메시지 신호 정규화 (-1 ~ 1 범위로)
    if np.max(np.abs(message)) > 0:
        normalized_message = message / np.max(np.abs(message))
    else:
        normalized_message = message
    
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    # AM 공식: s(t) = Ac[1 + ka*m(t)]cos(2πfct)
    return (1 + mod_index * normalized_message) * carrier

def frequency_modulation(t, message, carrier_freq, freq_deviation):
    """주파수 변조 (FM) - 정확한 적분 구현"""
    # 메시지 신호 정규화
    if np.max(np.abs(message)) > 0:
        normalized_message = message / np.max(np.abs(message))
    else:
        normalized_message = message
    
    # 트라페조이드 규칙을 사용한 정확한 적분
    dt = t[1] - t[0]
    integrated_message = np.zeros_like(t)
    for i in range(1, len(t)):
        integrated_message[i] = integrated_message[i-1] + (normalized_message[i] + normalized_message[i-1]) * dt / 2
    
    # FM 공식: s(t) = Ac*cos(2πfct + 2πkf∫m(τ)dτ)
    return np.cos(2 * np.pi * carrier_freq * t + 2 * np.pi * freq_deviation * integrated_message)

def phase_modulation(t, message, carrier_freq, phase_deviation):
    """위상 변조 (PM) - 완전한 구현"""
    # 메시지 신호 정규화
    if np.max(np.abs(message)) > 0:
        normalized_message = message / np.max(np.abs(message))
    else:
        normalized_message = message
    
    # PM 공식: s(t) = Ac*cos(2πfct + kp*m(t))
    return np.cos(2 * np.pi * carrier_freq * t + phase_deviation * normalized_message)

def amplitude_shift_keying(t, digital_data, carrier_freq, levels, bit_duration=0.25):
    """진폭편이변조 (ASK) - 다양한 레벨 지원"""
    signal = np.zeros_like(t)
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    
    for i, bit in enumerate(digital_data):
        start_time = i * bit_duration
        end_time = (i + 1) * bit_duration
        mask = (t >= start_time) & (t < end_time)
        
        # 비트에 따른 진폭 레벨 선택
        amplitude = levels[int(bit)] if bit < len(levels) else levels[-1]
        signal[mask] = amplitude * carrier[mask]
    
    return signal

def frequency_shift_keying(t, digital_data, carrier_freq, freq_deviation, bit_duration=0.25, continuous_phase=True):
    """주파수편이변조 (FSK) - 위상 연속/불연속 옵션"""
    signal = np.zeros_like(t)
    phase = 0  # 위상 연속성을 위한 누적 위상
    
    for i, bit in enumerate(digital_data):
        start_time = i * bit_duration
        end_time = (i + 1) * bit_duration
        mask = (t >= start_time) & (t < end_time)
        t_segment = t[mask] - start_time if continuous_phase else t[mask]
        
        # 비트에 따른 주파수 결정
        if bit == 1:
            freq = carrier_freq + freq_deviation
        else:
            freq = carrier_freq - freq_deviation
        
        if continuous_phase:
            # 위상 연속성 보장
            signal[mask] = np.cos(2 * np.pi * freq * t_segment + phase)
            # 다음 세그먼트를 위한 위상 업데이트
            if np.sum(mask) > 0:
                phase += 2 * np.pi * freq * bit_duration
                phase = phase % (2 * np.pi)
        else:
            # 위상 불연속 (각 비트마다 새로 시작)
            signal[mask] = np.cos(2 * np.pi * freq * t[mask])
    
    return signal

def phase_shift_keying(t, digital_data, carrier_freq, psk_type='BPSK', bit_duration=0.25):
    """위상편이변조 (PSK) - BPSK, QPSK 지원"""
    signal = np.zeros_like(t)
    
    if psk_type == 'BPSK':
        # Binary PSK (2진 PSK)
        for i, bit in enumerate(digital_data):
            start_time = i * bit_duration
            end_time = (i + 1) * bit_duration
            mask = (t >= start_time) & (t < end_time)
            
            if bit == 1:
                # 비트 1: 0도 위상
                signal[mask] = np.cos(2 * np.pi * carrier_freq * t[mask])
            else:
                # 비트 0: 180도 위상
                signal[mask] = np.cos(2 * np.pi * carrier_freq * t[mask] + np.pi)
    
    elif psk_type == 'QPSK':
        # Quadrature PSK (4진 PSK)
        symbol_duration = bit_duration * 2  # 2비트당 1심볼
        
        for i in range(0, len(digital_data)-1, 2):
            start_time = (i//2) * symbol_duration
            end_time = ((i//2) + 1) * symbol_duration
            mask = (t >= start_time) & (t < end_time)
            
            if i+1 < len(digital_data):
                # 2비트 조합에 따른 위상 결정
                bits = (digital_data[i], digital_data[i+1])
                
                if bits == (0, 0):      # 00 -> 45도
                    phase_shift = np.pi/4
                elif bits == (0, 1):    # 01 -> 135도  
                    phase_shift = 3*np.pi/4
                elif bits == (1, 1):    # 11 -> 225도
                    phase_shift = 5*np.pi/4
                else:                   # 10 -> 315도
                    phase_shift = 7*np.pi/4
                
                signal[mask] = np.cos(2 * np.pi * carrier_freq * t[mask] + phase_shift)
    
    return signal

def quadrature_amplitude_modulation(t, digital_data, carrier_freq, qam_type='4-QAM', bit_duration=0.25):
    """구상진폭변조 (QAM) - 4-QAM, 16-QAM 지원"""
    signal = np.zeros_like(t)
    
    if qam_type == '4-QAM':
        # 4-QAM (2비트당 1심볼)
        symbol_duration = bit_duration * 2
        
        for i in range(0, len(digital_data)-1, 2):
            start_time = (i//2) * symbol_duration
            end_time = ((i//2) + 1) * symbol_duration
            mask = (t >= start_time) & (t < end_time)
            
            if i+1 < len(digital_data):
                # 2비트 조합에 따른 I, Q 성분 결정 (Gray 코딩)
                bits = (digital_data[i], digital_data[i+1])
                
                if bits == (0, 0):      # 00 -> (+1, +1)
                    I, Q = 1, 1
                elif bits == (0, 1):    # 01 -> (+1, -1)
                    I, Q = 1, -1
                elif bits == (1, 1):    # 11 -> (-1, -1)
                    I, Q = -1, -1
                else:                   # 10 -> (-1, +1)
                    I, Q = -1, 1
                
                # QAM 신호: s(t) = I*cos(2πfct) - Q*sin(2πfct)
                carrier_cos = np.cos(2 * np.pi * carrier_freq * t[mask])
                carrier_sin = np.sin(2 * np.pi * carrier_freq * t[mask])
                signal[mask] = I * carrier_cos - Q * carrier_sin
    
    elif qam_type == '16-QAM':
        # 16-QAM (4비트당 1심볼)
        symbol_duration = bit_duration * 4
        
        for i in range(0, len(digital_data)-3, 4):
            start_time = (i//4) * symbol_duration
            end_time = ((i//4) + 1) * symbol_duration
            mask = (t >= start_time) & (t < end_time)
            
            if i+3 < len(digital_data):
                # 4비트 조합에 따른 I, Q 성분 결정
                bits = tuple(digital_data[i:i+4])
                
                # 16-QAM 성좌도 (Gray 코딩)
                constellation = {
                    (0,0,0,0): (-3,-3), (0,0,0,1): (-3,-1), (0,0,1,1): (-3,1), (0,0,1,0): (-3,3),
                    (0,1,0,0): (-1,-3), (0,1,0,1): (-1,-1), (0,1,1,1): (-1,1), (0,1,1,0): (-1,3),
                    (1,1,0,0): (1,-3),  (1,1,0,1): (1,-1),  (1,1,1,1): (1,1),  (1,1,1,0): (1,3),
                    (1,0,0,0): (3,-3),  (1,0,0,1): (3,-1),  (1,0,1,1): (3,1),  (1,0,1,0): (3,3)
                }
                
                I, Q = constellation.get(bits, (0, 0))
                
                # 정규화
                I, Q = I/3, Q/3
                
                # QAM 신호
                carrier_cos = np.cos(2 * np.pi * carrier_freq * t[mask])
                carrier_sin = np.sin(2 * np.pi * carrier_freq * t[mask])
                signal[mask] = I * carrier_cos - Q * carrier_sin
    
    return signal * 0.7  # 진폭 조정

# 변조된 신호 계산 함수
def calculate_modulated_signal(t, message, modulation_type, carrier_freq, mod_index):
    global digital_data, bit_duration, ask_levels, fsk_deviation, psk_type, qam_type
    
    if modulation_type == 'None':
        return message
    elif modulation_type == 'AM':
        return amplitude_modulation(t, message, carrier_freq, mod_index)
    elif modulation_type == 'FM':
        return frequency_modulation(t, message, carrier_freq, mod_index * 5)
    elif modulation_type == 'PM':
        return phase_modulation(t, message, carrier_freq, mod_index * np.pi)
    elif modulation_type == 'ASK':
        return amplitude_shift_keying(t, digital_data, carrier_freq, ask_levels, bit_duration)
    elif modulation_type == 'FSK':
        return frequency_shift_keying(t, digital_data, carrier_freq, fsk_deviation, bit_duration)
    elif modulation_type == 'PSK':
        return phase_shift_keying(t, digital_data, carrier_freq, psk_type, bit_duration)
    elif modulation_type == 'QAM':
        return quadrature_amplitude_modulation(t, digital_data, carrier_freq, qam_type, bit_duration)
    else:
        return message

# 디지털 데이터 생성 함수들
def generate_random_data(length=8):
    """랜덤 디지털 데이터 생성"""
    return np.random.randint(0, 2, length)

def generate_pattern_data(pattern='alternating', length=8):
    """패턴 기반 디지털 데이터 생성"""
    if pattern == 'alternating':
        return np.array([i % 2 for i in range(length)])
    elif pattern == 'all_ones':
        return np.ones(length, dtype=int)
    elif pattern == 'all_zeros':
        return np.zeros(length, dtype=int)
    elif pattern == 'manchester':
        return np.array([1, 0] * (length//2))[:length]
    else:
        return np.array([1, 0, 1, 1, 0, 1, 0, 0])[:length]

# 초기 사인파 및 합성파 계산
sine_waves = [calculate_sine(t, init_frequency[i], init_amplitude[i], init_phase[i], active[i]) for i in range(3)]
composite_wave = calculate_composite(t, init_frequency, init_amplitude, init_phase, active)

# 개별 사인파 + 합성파 그래프 (상단)
colors = ['red', 'green', 'blue', 'purple']
sine_lines = []
for i in range(3):
    line, = ax1.plot(t, sine_waves[i], color=colors[i], label=f'사인파 {i + 1}', alpha=0.7)
    sine_lines.append(line)

# 합성파 추가
composite_line_top, = ax1.plot(t, composite_wave, color=colors[3], linewidth=2, label='합성파')
sine_lines.append(composite_line_top)

ax1.set_xlabel('시간 (초)')
ax1.set_ylabel('진폭')
ax1.set_title('개별 사인파 + 합성파')
ax1.set_ylim(-3, 3)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 변조된 신호 그래프 (하단)
modulated_signal = calculate_modulated_signal(t, composite_wave, current_modulation, carrier_freq, modulation_index)
modulated_line, = ax2.plot(t, modulated_signal, color='black', linewidth=1.5)
ax2.set_xlabel('시간 (초)')
ax2.set_ylabel('진폭')
ax2.set_title('변조된 신호')
ax2.set_ylim(-3, 3)
ax2.grid(True, alpha=0.3)

# 주파수 슬라이더
ax_freq = []
freq_sliders = []
for i in range(3):
    ax_f = plt.axes([0.25, 0.45 - i * 0.04, 0.45, 0.025])
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
    ax_a = plt.axes([0.25, 0.33 - i * 0.04, 0.45, 0.025])
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
    ax_p = plt.axes([0.25, 0.21 - i * 0.04, 0.45, 0.025])
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

# 변조 파라미터 슬라이더
# 반송파 주파수 슬라이더
ax_carrier = plt.axes([0.25, 0.08, 0.45, 0.025])
carrier_slider = Slider(
    ax=ax_carrier,
    label='반송파 주파수 (Hz)',
    valmin=5,
    valmax=50,
    valinit=carrier_freq,
    color='orange'
)

# 변조 지수 슬라이더
ax_mod_index = plt.axes([0.25, 0.05, 0.45, 0.025])
mod_index_slider = Slider(
    ax=ax_mod_index,
    label='변조 지수',
    valmin=0,
    valmax=2.0,
    valinit=modulation_index,
    color='orange'
)

# 디지털 변조 세부 파라미터 슬라이더들 (기존 슬라이더 아래에 추가)
# 비트 지속시간 슬라이더
ax_bit_duration = plt.axes([0.75, 0.45, 0.2, 0.025])
bit_duration_slider = Slider(
    ax=ax_bit_duration,
    label='비트 지속시간',
    valmin=0.1,
    valmax=0.5,
    valinit=bit_duration,
    color='cyan'
)

# ASK 레벨 슬라이더들
ax_ask_low = plt.axes([0.75, 0.42, 0.2, 0.025])
ask_low_slider = Slider(
    ax=ax_ask_low,
    label='ASK 낮은 레벨',
    valmin=0,
    valmax=0.8,
    valinit=ask_levels[0],
    color='lightcoral'
)

ax_ask_high = plt.axes([0.75, 0.39, 0.2, 0.025])
ask_high_slider = Slider(
    ax=ax_ask_high,
    label='ASK 높은 레벨',
    valmin=0.2,
    valmax=2.0,
    valinit=ask_levels[1],
    color='lightcoral'
)

# FSK 주파수 편차 슬라이더
ax_fsk_dev = plt.axes([0.75, 0.36, 0.2, 0.025])
fsk_dev_slider = Slider(
    ax=ax_fsk_dev,
    label='FSK 주파수 편차',
    valmin=1,
    valmax=10,
    valinit=fsk_deviation,
    color='lightgreen'
)

# 활성화/비활성화 체크박스
rax = plt.axes([0.02, 0.7, 0.15, 0.2])
check_labels = [f'사인파 {i + 1}' for i in range(3)]
check = CheckButtons(
    ax=rax,
    labels=check_labels,
    actives=active
)

# 변조 방식 선택 라디오 버튼 (위치 조정)
rax_mod = plt.axes([0.75, 0.6, 0.2, 0.25])
modulation_types = ['None', 'AM', 'FM', 'PM', 'ASK', 'FSK', 'PSK', 'QAM']
radio = RadioButtons(rax_mod, modulation_types)

# PSK/QAM 타입 선택 라디오 버튼
rax_digital_type = plt.axes([0.75, 0.25, 0.2, 0.08])
digital_types = ['BPSK/4-QAM', 'QPSK/16-QAM']
radio_digital = RadioButtons(rax_digital_type, digital_types)

# 디지털 데이터 입력 방식 선택
rax_input = plt.axes([0.75, 0.15, 0.2, 0.08])
input_modes = ['Preset', 'Random', 'Pattern']
radio_input = RadioButtons(rax_input, input_modes)

# 디지털 데이터 표시 - 더 자세한 정보 포함
ax_digital = plt.axes([0.75, 0.05, 0.2, 0.08])

def update_digital_display():
    """디지털 데이터 표시 업데이트"""
    ax_digital.clear()
    try:
        ax_digital.text(0.1, 0.8, '디지털 데이터:', fontsize=9, weight='bold',
                       fontfamily=plt.rcParams['font.family'])
    except:
        ax_digital.text(0.1, 0.8, 'Digital Data:', fontsize=9, weight='bold')

    data_str = ''.join(map(str, digital_data))
    ax_digital.text(0.1, 0.5, data_str, fontsize=11, family='monospace')

    # 현재 설정 표시
    try:
        ax_digital.text(0.1, 0.2, f'길이: {len(digital_data)}비트, 지속: {bit_duration:.2f}s', 
                       fontsize=8, fontfamily=plt.rcParams['font.family'])
    except:
        ax_digital.text(0.1, 0.2, f'Length: {len(digital_data)}bits, Duration: {bit_duration:.2f}s', 
                       fontsize=8)

    ax_digital.set_xlim(0, 1)
    ax_digital.set_ylim(0, 1)
    ax_digital.axis('off')

# 초기 디지털 데이터 표시
update_digital_display()

# 업데이트 함수
def update(val=None):
    global current_modulation, carrier_freq, modulation_index, digital_data
    global bit_duration, ask_levels, fsk_deviation, psk_type, qam_type
    
    # 파라미터 가져오기
    freqs = [slider.val for slider in freq_sliders]
    amps = [slider.val for slider in amp_sliders]
    phases = [slider.val for slider in phase_sliders]
    carrier_freq = carrier_slider.val
    modulation_index = mod_index_slider.val
    
    # 디지털 변조 파라미터 업데이트
    bit_duration = bit_duration_slider.val
    ask_levels[0] = ask_low_slider.val
    ask_levels[1] = ask_high_slider.val
    fsk_deviation = fsk_dev_slider.val

    # 각 사인파 업데이트
    for i in range(3):
        sine_wave = calculate_sine(t, freqs[i], amps[i], phases[i], active[i])
        sine_lines[i].set_ydata(sine_wave)

    # 합성파 업데이트
    composite_wave = calculate_composite(t, freqs, amps, phases, active)
    sine_lines[3].set_ydata(composite_wave)  # 상단 그래프의 합성파

    # 변조된 신호 업데이트
    modulated_signal = calculate_modulated_signal(t, composite_wave, current_modulation, carrier_freq, modulation_index)
    modulated_line.set_ydata(modulated_signal)
    
    # Y축 범위 자동 조정
    if current_modulation in ['AM']:
        ax2.set_ylim(-3, 3)
    elif current_modulation in ['ASK']:
        max_level = max(ask_levels)
        ax2.set_ylim(-max_level*1.2, max_level*1.2)
    elif current_modulation in ['FSK', 'PSK', 'QAM']:
        ax2.set_ylim(-1.5, 1.5)
    elif current_modulation in ['FM', 'PM']:
        ax2.set_ylim(-1.2, 1.2)
    else:
        ax2.set_ylim(-3, 3)

    # 제목 업데이트 - 한글 폰트 적용
    try:
        if current_modulation == 'None':
            ax2.set_title('변조되지 않은 신호', fontfamily=plt.rcParams['font.family'])
        elif current_modulation in ['ASK', 'FSK', 'PSK', 'QAM']:
            detail_info = ""
            if current_modulation == 'ASK':
                detail_info = f" (레벨: {ask_levels[0]:.1f}/{ask_levels[1]:.1f})"
            elif current_modulation == 'FSK':
                detail_info = f" (편차: ±{fsk_deviation}Hz)"
            elif current_modulation == 'PSK':
                detail_info = f" ({psk_type})"
            elif current_modulation == 'QAM':
                detail_info = f" ({qam_type})"
            ax2.set_title(f'디지털 변조된 신호 - {current_modulation}{detail_info}', 
                         fontfamily=plt.rcParams['font.family'])
        else:
            ax2.set_title(f'아날로그 변조된 신호 - {current_modulation}', 
                         fontfamily=plt.rcParams['font.family'])
    except:
        ax2.set_title(f'Modulated Signal - {current_modulation}')
    
    # 디지털 데이터 표시 업데이트
    update_digital_display()
    
    fig.canvas.draw_idle()

# 체크박스 콜백 함수
def checkbox_callback(label):
    index = check_labels.index(label)
    active[index] = not active[index]
    update()

# 라디오 버튼 콜백 함수들
def radio_callback(label):
    global current_modulation
    current_modulation = label
    update()

def radio_digital_callback(label):
    global psk_type, qam_type
    if label == 'BPSK/4-QAM':
        psk_type = 'BPSK'
        qam_type = '4-QAM'
    else:  # 'QPSK/16-QAM'
        psk_type = 'QPSK'
        qam_type = '16-QAM'
    update()

def radio_input_callback(label):
    global digital_data, digital_input_mode
    digital_input_mode = label.lower()
    
    if label == 'Random':
        digital_data = generate_random_data(8)
    elif label == 'Pattern':
        # 순환하는 패턴들
        patterns = ['alternating', 'manchester', 'all_ones', 'all_zeros']
        pattern = patterns[np.random.randint(0, len(patterns))]
        digital_data = generate_pattern_data(pattern, 8)
    else:  # 'Preset'
        digital_data = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    
    update()

# 이벤트 연결
for slider in freq_sliders + amp_sliders + phase_sliders:
    slider.on_changed(update)
carrier_slider.on_changed(update)
mod_index_slider.on_changed(update)

# 디지털 변조 슬라이더들 연결
bit_duration_slider.on_changed(update)
ask_low_slider.on_changed(update)
ask_high_slider.on_changed(update)
fsk_dev_slider.on_changed(update)

check.on_clicked(checkbox_callback)
radio.on_clicked(radio_callback)
radio_digital.on_clicked(radio_digital_callback)
radio_input.on_clicked(radio_input_callback)

# 설명 텍스트 추가 - 더 자세한 디지털 변조 설명
try:
    fig.text(0.02, 0.02, '변조 방식 설명:\n아날로그: AM(진폭), FM(주파수), PM(위상)\n디지털: ASK(진폭편이), FSK(주파수편이), PSK(위상편이), QAM(구상진폭)\n\n디지털 변조 세부사항:\n• ASK: 비트에 따라 진폭 변경 (0비트/1비트 레벨 조절 가능)\n• FSK: 비트에 따라 주파수 변경 (편차 조절 가능)\n• PSK: BPSK(2진) 또는 QPSK(4진) 선택 가능\n• QAM: 4-QAM(2비트/심볼) 또는 16-QAM(4비트/심볼)', 
             fontsize=7, verticalalignment='bottom', 
             fontfamily=plt.rcParams['font.family'])
except:
    # 폰트 설정 실패 시 기본 텍스트
    fig.text(0.02, 0.02, 'Modulation Types:\nAnalog: AM, FM, PM\nDigital: ASK, FSK, PSK, QAM\n\nDigital Details:\n• ASK: Amplitude levels adjustable\n• FSK: Frequency deviation adjustable\n• PSK: BPSK or QPSK selectable\n• QAM: 4-QAM or 16-QAM selectable', 
             fontsize=7, verticalalignment='bottom')

plt.show()