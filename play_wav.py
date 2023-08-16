import simpleaudio
wave_obj = simpleaudio.WaveObject.from_wave_file('PinkPanther30.wav')
play_obj = wave_obj.play()
play_obj.wait_done()
