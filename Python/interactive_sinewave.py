import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

# 초기 파라미터 설정
init_frequency = [1, 2, 3]  # 초기 주파수 (Hz)
init_amplitude = [1, 0.5, 0.3]  # 초기 진폭
init_phase = [0, 0, 0]  # 초기 위상 (라디안)
active = [True, True, True]  # 활성화 상태
t = np.linspace(0, 2, 1000)  # 0부터 2초까지 1000개 샘플

# 그래프 초기화
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(left=0.25, bottom=0.5)


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

# 활성화/비활성화 체크박스
rax = plt.axes([0.05, 0.7, 0.15, 0.15])
check_labels = [f'사인파 {i + 1}' for i in range(3)]
check = CheckButtons(
    ax=rax,
    labels=check_labels,
    actives=active
)


# 업데이트 함수
def update(val):
    # 각 사인파의 파라미터 가져오기
    freqs = [slider.val for slider in freq_sliders]
    amps = [slider.val for slider in amp_sliders]
    phases = [slider.val for slider in phase_sliders]

    # 각 사인파 업데이트
    for i in range(3):
        sine_wave = calculate_sine(t, freqs[i], amps[i], phases[i], active[i])
        sine_lines[i].set_ydata(sine_wave)

    # 합성파 업데이트
    composite_wave = calculate_composite(t, freqs, amps, phases, active)
    composite_line.set_ydata(composite_wave)

    fig.canvas.draw_idle()


# 체크박스 콜백 함수
def checkbox_callback(label):
    index = check_labels.index(label)
    active[index] = not active[index]
    update(None)


# 이벤트 연결
for slider in freq_sliders + amp_sliders + phase_sliders:
    slider.on_changed(update)
check.on_clicked(checkbox_callback)

plt.show()