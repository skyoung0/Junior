import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
import matplotlib.patches as patches
import platform
import matplotlib.font_manager as fm

# 5/26 - 과제 시작.. 한글 폰트 문제
# 예제 코드 분석 부터
def setup_korean_font():
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        korean_fonts = ['AppleGothic', 'Arial Unicode MS', 'AppleSDGothicNeo-Regular']
    elif system == 'Windows':  # Windows
        korean_fonts = ['Malgun Gothic', 'Microsoft YaHei', 'SimHei']
    else:  # Linux 및 기타
        korean_fonts = ['DejaVu Sans', 'Liberation Sans', 'Arial']
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in korean_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            break
    else:
        print("Warning: 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

# 5/26 저녁 - 기본 파라미터들 설정

init_frequency = [1, 2, 3]  
init_amplitude = [1, 0.5, 0.3]  
init_phase = [0, 0, 0]  
active = [True, True, True]  
t = np.linspace(0, 2, 1000)  # 2초면 충분할까? 나중에 늘려야 할 수도

# 5/27 - 변조 관련 변수들 추가

carrier_freq = 10  
modulation_index = 0.5  
digital_data = np.array([1, 0, 1, 1, 0, 1, 0, 0])  # 임의로 설정
current_modulation = 'None'  

# 5/28 - 디지털 변조 파라미터들 생각보다 복잡
bit_duration = 0.25  
ask_levels = [0.2, 1.0]  
fsk_deviation = 3  
psk_type = 'BPSK'  
qam_type = '4-QAM'  
digital_input_mode = 'preset'  

# 5/28 밤 12시 - 그래프 레이아웃 잡기... subplot 위치 조정
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(left=0.25, bottom=0.55, right=0.95, top=0.95)

def calculate_sine(t, freq, amp, phase, is_active):
    if is_active:
        return amp * np.sin(2 * np.pi * freq * t + phase)
    else:
        return np.zeros_like(t)

def calculate_composite(t, freqs, amps, phases, actives):
    composite = np.zeros_like(t)
    for i in range(3):
        composite += calculate_sine(t, freqs[i], amps[i], phases[i], actives[i])
    return composite

def generate_digital_signal(t, data, bit_duration=0.25):
    signal = np.zeros_like(t)
    for i, bit in enumerate(data):
        start_time = i * bit_duration
        end_time = (i + 1) * bit_duration
        mask = (t >= start_time) & (t < end_time)
        signal[mask] = bit
    return signal

# 5/29 오전 - AM 변조 구현. 수식 틀
# s(t) = Ac[1 + ka*m(t)]cos(2πfct) 
def amplitude_modulation(t, message, carrier_freq, mod_index):
    if np.max(np.abs(message)) > 0:
        normalized_message = message / np.max(np.abs(message))
    else:
        normalized_message = message
    
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    return (1 + mod_index * normalized_message) * carrier

# 5/29 오후 - FM이 제일 어려움... 적분 어떻게 구현할지 고민
# claude가 트라페조이드 규칙 써야 한다고함
def frequency_modulation(t, message, carrier_freq, freq_deviation):
    if np.max(np.abs(message)) > 0:
        normalized_message = message / np.max(np.abs(message))
    else:
        normalized_message = message
    
    # 5/29 저녁 - 이 부분. 적분 구현
    dt = t[1] - t[0]
    integrated_message = np.zeros_like(t)
    for i in range(1, len(t)):
        integrated_message[i] = integrated_message[i-1] + (normalized_message[i] + normalized_message[i-1]) * dt / 2
    
    return np.cos(2 * np.pi * carrier_freq * t + 2 * np.pi * freq_deviation * integrated_message)

# 5/30 - PM은 AM보다 쉬움. 그냥 위상만 바꾸면 됨
def phase_modulation(t, message, carrier_freq, phase_deviation):
    if np.max(np.abs(message)) > 0:
        normalized_message = message / np.max(np.abs(message))
    else:
        normalized_message = message
    
    return np.cos(2 * np.pi * carrier_freq * t + phase_deviation * normalized_message)

# 5/30 저녁 - 디지털 변조 시작. ASK부터
# 그냥 진폭만 바꾸면 됨
def amplitude_shift_keying(t, digital_data, carrier_freq, levels, bit_duration=0.25):
    signal = np.zeros_like(t)
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    
    for i, bit in enumerate(digital_data):
        start_time = i * bit_duration
        end_time = (i + 1) * bit_duration
        mask = (t >= start_time) & (t < end_time)
        
        amplitude = levels[int(bit)] if bit < len(levels) else levels[-1]
        signal[mask] = amplitude * carrier[mask]
    
    return signal

# 5/31 - FSK... 위상 연속성 문제
# continuous_phase 안 하면 신호가 이상하게 나옴
def frequency_shift_keying(t, digital_data, carrier_freq, freq_deviation, bit_duration=0.25, continuous_phase=True):
    signal = np.zeros_like(t)
    phase = 0  
    
    for i, bit in enumerate(digital_data):
        start_time = i * bit_duration
        end_time = (i + 1) * bit_duration
        mask = (t >= start_time) & (t < end_time)
        t_segment = t[mask] - start_time if continuous_phase else t[mask]
        
        if bit == 1:
            freq = carrier_freq + freq_deviation
        else:
            freq = carrier_freq - freq_deviation
        
        if continuous_phase:
            signal[mask] = np.cos(2 * np.pi * freq * t_segment + phase)
            if np.sum(mask) > 0:
                phase += 2 * np.pi * freq * bit_duration
                phase = phase % (2 * np.pi)
        else:
            signal[mask] = np.cos(2 * np.pi * freq * t[mask])
    
    return signal

# 6/1 - PSK. BPSK는 쉬운데 QPSK는 머리 아픔
# 2비트씩 묶어서 처리해야 함. 00, 01, 11, 10 -> 4가지 위상
def phase_shift_keying(t, digital_data, carrier_freq, psk_type='BPSK', bit_duration=0.25):
    signal = np.zeros_like(t)
    
    if psk_type == 'BPSK':
        for i, bit in enumerate(digital_data):
            start_time = i * bit_duration
            end_time = (i + 1) * bit_duration
            mask = (t >= start_time) & (t < end_time)
            
            if bit == 1:
                signal[mask] = np.cos(2 * np.pi * carrier_freq * t[mask])
            else:
                signal[mask] = np.cos(2 * np.pi * carrier_freq * t[mask] + np.pi)
    
    elif psk_type == 'QPSK':
        # 6/1 밤 - QPSK 구현. claude 보면서 함
        symbol_duration = bit_duration * 2  
        
        for i in range(0, len(digital_data)-1, 2):
            start_time = (i//2) * symbol_duration
            end_time = ((i//2) + 1) * symbol_duration
            mask = (t >= start_time) & (t < end_time)
            
            if i+1 < len(digital_data):
                bits = (digital_data[i], digital_data[i+1])
                
                if bits == (0, 0):      
                    phase_shift = np.pi/4
                elif bits == (0, 1):    
                    phase_shift = 3*np.pi/4
                elif bits == (1, 1):    
                    phase_shift = 5*np.pi/4
                else:                   
                    phase_shift = 7*np.pi/4
                
                signal[mask] = np.cos(2 * np.pi * carrier_freq * t[mask] + phase_shift)
    
    return signal

# 6/2 - QAM. 제일 어려움. I, Q 성분 따로 계산해야 함
# 16-QAM은 진짜 복잡. 4x4 constellation 그려가면서 함
def quadrature_amplitude_modulation(t, digital_data, carrier_freq, qam_type='4-QAM', bit_duration=0.25):
    signal = np.zeros_like(t)
    
    if qam_type == '4-QAM':
        symbol_duration = bit_duration * 2
        
        for i in range(0, len(digital_data)-1, 2):
            start_time = (i//2) * symbol_duration
            end_time = ((i//2) + 1) * symbol_duration
            mask = (t >= start_time) & (t < end_time)
            
            if i+1 < len(digital_data):
                bits = (digital_data[i], digital_data[i+1])
                
                # 6/2 오후 - Gray 코딩 적용. 에러 최소화
                if bits == (0, 0):      
                    I, Q = 1, 1
                elif bits == (0, 1):    
                    I, Q = 1, -1
                elif bits == (1, 1):    
                    I, Q = -1, -1
                else:                   
                    I, Q = -1, 1
                
                carrier_cos = np.cos(2 * np.pi * carrier_freq * t[mask])
                carrier_sin = np.sin(2 * np.pi * carrier_freq * t[mask])
                signal[mask] = I * carrier_cos - Q * carrier_sin
    
    elif qam_type == '16-QAM':
        # 6/2 밤 - 16-QAM. 4비트씩 처리. 16가지 경우의 수
        symbol_duration = bit_duration * 4
        
        for i in range(0, len(digital_data)-3, 4):
            start_time = (i//4) * symbol_duration
            end_time = ((i//4) + 1) * symbol_duration
            mask = (t >= start_time) & (t < end_time)
            
            if i+3 < len(digital_data):
                bits = tuple(digital_data[i:i+4])
                
                # 6/3 새벽 - 이 constellation 표
                constellation = {
                    (0,0,0,0): (-3,-3), (0,0,0,1): (-3,-1), (0,0,1,1): (-3,1), (0,0,1,0): (-3,3),
                    (0,1,0,0): (-1,-3), (0,1,0,1): (-1,-1), (0,1,1,1): (-1,1), (0,1,1,0): (-1,3),
                    (1,1,0,0): (1,-3),  (1,1,0,1): (1,-1),  (1,1,1,1): (1,1),  (1,1,1,0): (1,3),
                    (1,0,0,0): (3,-3),  (1,0,0,1): (3,-1),  (1,0,1,1): (3,1),  (1,0,1,0): (3,3)
                }
                
                I, Q = constellation.get(bits, (0, 0))
                
                I, Q = I/3, Q/3
                
                carrier_cos = np.cos(2 * np.pi * carrier_freq * t[mask])
                carrier_sin = np.sin(2 * np.pi * carrier_freq * t[mask])
                signal[mask] = I * carrier_cos - Q * carrier_sin
    
    return signal * 0.7  # 6/3 - 진폭 조정. 너무 크면 보기 안 좋음

# 6/3 오전 - 변조 통합 함수. 모든 변조 방식을 하나로
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

# 6/3 오후 - 디지털 데이터 생성 함수들 추가
# 사용자가 랜덤이나 패턴으로도 선택할 수 있게
def generate_random_data(length=8):
    return np.random.randint(0, 2, length)

def generate_pattern_data(pattern='alternating', length=8):
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

# 6/3 저녁 - 초기 계산들
sine_waves = [calculate_sine(t, init_frequency[i], init_amplitude[i], init_phase[i], active[i]) for i in range(3)]
composite_wave = calculate_composite(t, init_frequency, init_amplitude, init_phase, active)

# 그래프 그리기 시작
colors = ['red', 'green', 'blue', 'purple']
sine_lines = []
for i in range(3):
    line, = ax1.plot(t, sine_waves[i], color=colors[i], label=f'사인파 {i + 1}', alpha=0.7)
    sine_lines.append(line)

composite_line_top, = ax1.plot(t, composite_wave, color=colors[3], linewidth=2, label='합성파')
sine_lines.append(composite_line_top)

ax1.set_xlabel('시간 (초)')
ax1.set_ylabel('진폭')
ax1.set_title('개별 사인파 + 합성파')
ax1.set_ylim(-3, 3)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

modulated_signal = calculate_modulated_signal(t, composite_wave, current_modulation, carrier_freq, modulation_index)
modulated_line, = ax2.plot(t, modulated_signal, color='black', linewidth=1.5)
ax2.set_xlabel('시간 (초)')
ax2.set_ylabel('진폭')
ax2.set_title('변조된 신호')
ax2.set_ylim(-3, 3)
ax2.grid(True, alpha=0.3)

# 6/3 밤 - 슬라이더들 배치. 노가다
# 위치 잡는데 30분은 걸린 듯
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

# 6/4 새벽 - 변조 파라미터 슬라이더들
ax_carrier = plt.axes([0.25, 0.08, 0.45, 0.025])
carrier_slider = Slider(
    ax=ax_carrier,
    label='반송파 주파수 (Hz)',
    valmin=5,
    valmax=50,
    valinit=carrier_freq,
    color='orange'
)

ax_mod_index = plt.axes([0.25, 0.05, 0.45, 0.025])
mod_index_slider = Slider(
    ax=ax_mod_index,
    label='변조 지수',
    valmin=0,
    valmax=2.0,
    valinit=modulation_index,
    color='orange'
)

# 6/4 오전 - 디지털 변조용 슬라이더들 추가
# 오른쪽에 배치. 공간 부족해서 크기 줄임
ax_bit_duration = plt.axes([0.75, 0.45, 0.2, 0.025])
bit_duration_slider = Slider(
    ax=ax_bit_duration,
    label='비트 지속시간',
    valmin=0.1,
    valmax=0.5,
    valinit=bit_duration,
    color='cyan'
)

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

ax_fsk_dev = plt.axes([0.75, 0.36, 0.2, 0.025])
fsk_dev_slider = Slider(
    ax=ax_fsk_dev,
    label='FSK 주파수 편차',
    valmin=1,
    valmax=10,
    valinit=fsk_deviation,
    color='lightgreen'
)

# 6/4 오후 - 체크박스와 라디오 버튼들
# 활성화/비활성화 기능 추가
rax = plt.axes([0.02, 0.7, 0.15, 0.2])
check_labels = [f'사인파 {i + 1}' for i in range(3)]
check = CheckButtons(
    ax=rax,
    labels=check_labels,
    actives=active
)

# 변조 방식 선택 라디오 버튼
rax_mod = plt.axes([0.75, 0.6, 0.2, 0.25])
modulation_types = ['None', 'AM', 'FM', 'PM', 'ASK', 'FSK', 'PSK', 'QAM']
radio = RadioButtons(rax_mod, modulation_types)

rax_digital_type = plt.axes([0.75, 0.25, 0.2, 0.08])
digital_types = ['BPSK/4-QAM', 'QPSK/16-QAM']
radio_digital = RadioButtons(rax_digital_type, digital_types)

rax_input = plt.axes([0.75, 0.15, 0.2, 0.08])
input_modes = ['Preset', 'Random', 'Pattern']
radio_input = RadioButtons(rax_input, input_modes)

# 6/4 저녁 - 디지털 데이터 표시부
ax_digital = plt.axes([0.75, 0.05, 0.2, 0.08])

def update_digital_display():
    ax_digital.clear()
    try:
        ax_digital.text(0.1, 0.8, '디지털 데이터:', fontsize=9, weight='bold',
                       fontfamily=plt.rcParams['font.family'])
    except:
        ax_digital.text(0.1, 0.8, 'Digital Data:', fontsize=9, weight='bold')

    data_str = ''.join(map(str, digital_data))
    ax_digital.text(0.1, 0.5, data_str, fontsize=11, family='monospace')

    try:
        ax_digital.text(0.1, 0.2, f'길이: {len(digital_data)}비트, 지속: {bit_duration:.2f}s', 
                       fontsize=8, fontfamily=plt.rcParams['font.family'])
    except:
        ax_digital.text(0.1, 0.2, f'Length: {len(digital_data)}bits, Duration: {bit_duration:.2f}s', 
                       fontsize=8)

    ax_digital.set_xlim(0, 1)
    ax_digital.set_ylim(0, 1)
    ax_digital.axis('off')

update_digital_display()

# 6/4 밤 - 메인 업데이트 함수.
# 모든 파라미터 변경사항을 여기서 처리해야 함
def update(val=None):
    global current_modulation, carrier_freq, modulation_index, digital_data
    global bit_duration, ask_levels, fsk_deviation, psk_type, qam_type
    
    freqs = [slider.val for slider in freq_sliders]
    amps = [slider.val for slider in amp_sliders]
    phases = [slider.val for slider in phase_sliders]
    carrier_freq = carrier_slider.val
    modulation_index = mod_index_slider.val
    
    bit_duration = bit_duration_slider.val
    ask_levels[0] = ask_low_slider.val
    ask_levels[1] = ask_high_slider.val
    fsk_deviation = fsk_dev_slider.val

    for i in range(3):
        sine_wave = calculate_sine(t, freqs[i], amps[i], phases[i], active[i])
        sine_lines[i].set_ydata(sine_wave)

    composite_wave = calculate_composite(t, freqs, amps, phases, active)
    sine_lines[3].set_ydata(composite_wave)  

    modulated_signal = calculate_modulated_signal(t, composite_wave, current_modulation, carrier_freq, modulation_index)
    modulated_line.set_ydata(modulated_signal)
    
    # Y축 범위 자동 조정 - 각 변조 방식마다 다르게
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

    # 제목 업데이트 - 각 변조 방식의 상세 정보 표시
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
    
    update_digital_display()
    
    fig.canvas.draw_idle()

# 6/4 밤 11시 - 콜백 함수들. 이벤트 처리
def checkbox_callback(label):
    index = check_labels.index(label)
    active[index] = not active[index]
    update()

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
        # 6/4 - 패턴 종류 랜덤으로 선택하게 함
        patterns = ['alternating', 'manchester', 'all_ones', 'all_zeros']
        pattern = patterns[np.random.randint(0, len(patterns))]
        digital_data = generate_pattern_data(pattern, 8)
    else:  # 'Preset'
        digital_data = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    
    update()

# 6/4 밤 12시 - 이벤트 연결. 마지막
for slider in freq_sliders + amp_sliders + phase_sliders:
    slider.on_changed(update)
carrier_slider.on_changed(update)
mod_index_slider.on_changed(update)

bit_duration_slider.on_changed(update)
ask_low_slider.on_changed(update)
ask_high_slider.on_changed(update)
fsk_dev_slider.on_changed(update)

check.on_clicked(checkbox_callback)
radio.on_clicked(radio_callback)
radio_digital.on_clicked(radio_digital_callback)
radio_input.on_clicked(radio_input_callback)

# 6/4 마지막 - 설명 텍스트 추가 
# 근데 공간이 부족해서 작은 글씨로 구겨넣음
try:
    fig.text(0.02, 0.02, '변조 방식 설명:\n아날로그: AM(진폭), FM(주파수), PM(위상)\n디지털: ASK(진폭편이), FSK(주파수편이), PSK(위상편이), QAM(구상진폭)\n\n디지털 변조 세부사항:\n• ASK: 비트에 따라 진폭 변경 (0비트/1비트 레벨 조절 가능)\n• FSK: 비트에 따라 주파수 변경 (편차 조절 가능)\n• PSK: BPSK(2진) 또는 QPSK(4진) 선택 가능\n• QAM: 4-QAM(2비트/심볼) 또는 16-QAM(4비트/심볼)', 
             fontsize=7, verticalalignment='bottom', 
             fontfamily=plt.rcParams['font.family'])
except:
    # 6/4 새벽 2시 - 한글 폰트 없는 환경 대비
    fig.text(0.02, 0.02, 'Modulation Types:\nAnalog: AM, FM, PM\nDigital: ASK, FSK, PSK, QAM\n\nDigital Details:\n• ASK: Amplitude levels adjustable\n• FSK: Frequency deviation adjustable\n• PSK: BPSK or QPSK selectable\n• QAM: 4-QAM or 16-QAM selectable', 
             fontsize=7, verticalalignment='bottom')

# 6/4 새벽 3시 - 드디어 완성!!!
# 테스트 좀 더 해보고 제출
plt.show()