from typing import List

import tqdm

from pyalp.beams import beams
from pyalp.components import phase_screen

#==============================================================================
#==============================================================================
class Channel:
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(
      self,
      turb: phase_screen.Kolmogorov,
      distance: float,
      phase_screen_count: int):
        self.screen_distance = distance / phase_screen_count
        self.phase_screens = []
        for _ in range(phase_screen_count):
            screen = turb.get_phase_screen(self.screen_distance)
            self.phase_screens.append(screen)
        self.reverse_phase_screens = list(reversed(self.phase_screens))

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def forward(self, beam: beams.Gaussian, progress_bar: bool=False):
        self.__single_pass(
            beam,
            self.phase_screens,
            progress_bar)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def backward(self, beam: beams.Gaussian, progress_bar: bool=False):
        self.__single_pass(
            beam,
            self.reverse_phase_screens,
            progress_bar)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def __single_pass(
    self,
    beam: beams.Gaussian,
    phase_screens:List[phase_screen.Kolmogorov],
    progress_bar: bool):
        disable_pbar = not progress_bar
        for phase_screen in tqdm.tqdm(phase_screens, disable=disable_pbar):
            beam.propagate(self.screen_distance / 2.0)
            beam.distort(phase=phase_screen)
            beam.propagate(self.screen_distance / 2.0)
