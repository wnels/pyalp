import numpy as np
import tqdm

import turbulence

#==============================================================================
#==============================================================================
class atm_channel:
    def __init__(self, turb, distance, phase_screen_count):
        self.screen_distance = distance / phase_screen_count
        self.phase_screens = []
        for index in range(phase_screen_count):
            screen = turb.get_phase_screen(self.screen_distance)
            self.phase_screens.append(screen)

    def forward(self, beam):
        for phase_screen in tqdm.tqdm(self.phase_screens):
            beam.propagate(self.screen_distance / 2.0)
            beam.distort(phase_screen)
            beam.propagate(self.screen_distance / 2.0)
