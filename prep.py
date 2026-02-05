import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "blues.00000.wav"

#waveform
# 우리가 지난번에 배웠듯이 sr (단위가 Hz)만큼 초당 n가의 점을 시간 축 위에 찍으면
# 해당 점에 대응하는 아날로그 신호를 quantize 해서 대략적인 값을 저장하는것이
# ADC(analog digital conversion)이다
# signal은 각 sample에 대응하는 quantize된 값들을 저장하는 배열이고
# sr은 말 그대로 sample rate다
# 즉 우리가 다루는 음원이 30초 짜리니까 sr=22050이면 총 22050*30개의 samle이 생기고
# 배열인 signal 역시 각 샘플에 대응하는 값들이 저장되므로
# 22050 * 30개의 값이 저장된다
signal, sr = librosa.load(file, sr=22050) #sr * T -> 22050 * 30
    #왜 하필 22050인가 -> 일반적으로 인간이 들을 수 있는 가청 주파수가 20000hz이고
    #나이퀴스트 정리에 의하면 어떤 주파수를 제대로 기록하려면, 그 주파수보다 최소
    #2배 더 빠르게 점을 찍어야 한다
    # 그게 일반적인 디지털 음원이 44100hz를 사용하는 이유인데
    # 계산량을 줄이기 위해 딥러닝에서 일반적으로 22050hz를 sr로서 자주 이용한다
# print(f"signal 타입: {type(signal)}")
# print(f"signal의 처음 5개 값: {signal[:5]}") # -1~1 사이의 소수들
# print(f"데이터 전체 개수: {len(signal)}")
# print("-" * 30)
# print(f"sr 타입: {type(sr)}")
# print(f"설정된 SR 값: {sr}")
    # signal과 sr에는 각각 다음과 같은 값들이 저장된다
    # signal 타입: <class 'numpy.ndarray'>
    # signal의 처음 5개 값: [ 0.00732422  0.01660156  0.00762939 -0.00350952 -0.0022583 ]
    # 데이터 전체 개수: 661794 -> 소스점 단위까지 계산하므로 정확히 나누어 떨이지는 않음
    # ------------------------------
    # sr 타입: <class 'int'>
    # 설정된 SR 값: 22050
# librosa.display.waveshow(signal, sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Aplitude")
# plt.show()

# fft -> spectrum
fft = np.fft.fft(signal)

# 앞서 배웠듯이 magnitude는 각 주파수가 얼만큼 전체 소리에 기여하는지 나타냄
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]

plt.plot(left_frequency, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()
# stft -> spectogram
# MFCCs