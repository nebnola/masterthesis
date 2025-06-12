from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Optional

import plotnine as p9
from plotnine import ggplot

theme_chill = (
        p9.theme_bw() +
        p9.theme(
            plot_background=p9.element_rect(fill=(0, 0, 0, 0), size=0),
            panel_background=p9.element_rect(fill=(0, 0, 0, 0)),
            legend_background=p9.element_rect(fill=(0, 0, 0, 0)),
        ))

theme_arguelles = (
        p9.theme_bw(base_size=14) +
        p9.theme(
            text=p9.element_text(family="Alegreya", color="#343A40"),
            axis_text=p9.element_text(family="sans"),
            plot_background=p9.element_rect(fill=(0, 0, 0, 0), size=0),
            panel_background=p9.element_rect(fill=(0, 0, 0, 0)),
            legend_background=p9.element_rect(fill=(0, 0, 0, 0)),
            strip_background=p9.element_rect(fill="#343A40"),
            strip_text=p9.element_text(color="#F8F9FA"),
        )
)

theme_arguelles_sans = (
        p9.theme_bw(base_size=14) +
        p9.theme(
            text=p9.element_text(family="Alegreya Sans", color="#343A40"),
            axis_text=p9.element_text(family="sans", size=9, color="#4D4D4D"),
            plot_background=p9.element_rect(fill=(0, 0, 0, 0), size=0),
            panel_background=p9.element_rect(fill=(0, 0, 0, 0)),
            legend_background=p9.element_rect(fill=(0, 0, 0, 0)),
            strip_background=p9.element_rect(fill="#343A40"),
            strip_text=p9.element_text(color="#F8F9FA"),
        )
)

theme_thesis = (
    p9.theme_bw(base_size=10.95, base_family="P052") + # Palatino
    p9.theme(
        strip_background=p9.element_rect(fill="white"),
        figure_size=(6,4),
    )
)


@dataclass
class MuxProfile:
    """A MuxProfile instance specifies a profile for export of ggplot. See PlotMux"""
    path: Path | str
    theme: p9.theme

class PlotMux:
    profiles: dict[str, MuxProfile] # A label: MuxProfile dict containing the different profiles
    default_profile: str # The label of the default profile, which is used for in-line display in notebooks

    def __init__(self, plot: ggplot, all: Optional[p9.theme] = None, **kwargs):
        """
        Create a new PlotMux (MUltieXport) object, allowing to save a plot according to different profiles.
        A profile specifies the directory to save to and a theme to apply.
        Args:
            plot: The `plotnine.ggplot` to be used. It WILL be modified by the themes.
            all: The theme which is applied for all profiles (optional)
            **kwargs: The themes which are applied to individual profiles. Use the profile label as key and a
            `plotnine.theme` as values (optional)
        """
        self.plot = plot
        self.theme_all = all or p9.theme()
        self.themes_profile = kwargs

    def render(self, profile: str) -> ggplot:
        """
        Add theming to the plot according to a specified profile
        """
        theme = self.profiles[profile].theme
        p = self.plot + theme + self.theme_all
        if profile in self.themes_profile:
            p += self.themes_profile[profile]
        return p

    def _ipython_display_(self):
        self.show()

    def show(self) -> None:
        p = self.render(self.default_profile)
        p.show()

    def save(self, filename: str, figure_sizes: Optional[dict[str | tuple[Number, Number]]] = None, verbose=False, **kwargs) -> None:
        """
        Save plot according to all profiles
        Args:
            filename: Filename used for saving. The directories are specified in `PlotMux.profiles`
            figure_sizes: A dictionary containing the sizes to be used for saving for each profile
            verbose: Passed to `ggplot.save`
            **kwargs: More keyword arguments to be passed to `ggplot.save`
        """
        if figure_sizes is None:
            figure_sizes = {}

        for name, profile in self.profiles.items():
            p = self.render(name)
            try:
                width, height = figure_sizes[name]
            except KeyError:
                width, height = None, None
            p.save(profile.path / filename, width= width, height=height, verbose=verbose, **kwargs)


class MaMux(PlotMux):
    """
    PlotMux class used for this master thesis
    """
    projectroot=Path(__file__).parent.parent
    profiles = {
        "prez": MuxProfile(projectroot/"plots"/"prez", theme_arguelles_sans),
        "thesis": MuxProfile(projectroot/"plots"/"thesis", theme_thesis)
    }
    default_profile = "prez"

    def save(self,
             filename: str,
             thesis_size: Optional[tuple[Number, Number]] = None,
             figure_sizes: Optional[dict[str | tuple[Number, Number]]] = None,
             verbose=False,
             **kwargs) -> None:

        if figure_sizes is None:
            figure_sizes = {}
        if thesis_size is not None:
            figure_sizes["thesis"] = thesis_size
        super().save(filename, figure_sizes, verbose, **kwargs)
