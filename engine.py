import numpy as np
import tqdm

#==============================================================================
#==============================================================================
class atm_channel:
    def __init__(self, turb, distance, phase_screen_count):
        self.screen_distance = distance / phase_screen_count
        self.phase_screens = []
        for index in range(phase_screen_count):
            screen = turb.get_phase_screen(self.screen_distance)
            self.phase_screens.append(screen)

    def forward(self, beam, progress_bar=False):
        self.__single_pass(
            beam,
            self.phase_screens,
            progress_bar)

    def backward(self, beam, progress_bar=False):
        self.__single_pass(
            beam,
            self.phase_screens.reverse(),
            progress_bar)

    def __single_pass(self, beam, phase_screens, progress_bar):
        disable_pbar = not progress_bar
        for phase_screen in tqdm.tqdm(phase_screens, disable=disable_pbar):
            beam.propagate(self.screen_distance / 2.0)
            beam.distort(phase_screen)
            beam.propagate(self.screen_distance / 2.0)