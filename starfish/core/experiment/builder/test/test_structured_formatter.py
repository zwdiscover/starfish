import csv
import dataclasses
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, MutableSequence, Optional, Sequence

import numpy as np
import pytest
from slicedimage import ImageFormat

from starfish.core.experiment.experiment import Experiment, FieldOfView
from starfish.core.imagestack.test.imagestack_test_utils import verify_physical_coordinates
from starfish.core.types import Axes, Coordinates, CoordinateValue
from .factories import unique_data
from .. import (
    FetchedTile,
    tile_fetcher_factory,
    TileFetcher,
    TileIdentifier,
)
from ..structured_formatter import (
    TILE_COORDINATE_NAMES,
)


@pytest.mark.parametrize(
    ["tile_format", "in_place"],
    [
        (ImageFormat.NUMPY, True),
        (ImageFormat.NUMPY, False),
        (ImageFormat.TIFF, True),
        (ImageFormat.TIFF, False),
    ]
)
def test_single_aligned_regular_fov(
        tmpdir,
        tile_format: ImageFormat,
        in_place: bool,
        rounds=(1, 2, 4),
        chs=(2, 3, 4),
        zplanes=(0, 1, 2),
        tile_height=100,
        tile_width=60,
        x_coords=(0.0, 0.1),
        y_coords=(0.1, 0.2),
        zplane_to_coords={0: 0.20, 1: 0.25, 2: 0.3},
):
    """Write the tiles for a single aligned (physical coordinates) regular (the dimensions have the
    same cardinality) FOV.  Then build an experiment from the tiles.  Finally, load the resulting
    experiment as an ImageStack and verify that the data matches."""
    tempdir_path: Path = Path(tmpdir)
    tile_identifiers: Sequence[TileIdentifier] = [
        TileIdentifier(0, round_label, ch_label, zplane_label)
        for round_label in rounds
        for ch_label in chs
        for zplane_label in zplanes
    ]
    tile_fetcher: TileFetcher = tile_fetcher_factory(
        UniqueTiles,
        pass_tile_indices=True,
        fovs=[0],
        rounds=rounds,
        chs=chs,
        zplanes=zplanes,
        tile_height=tile_height,
        tile_width=tile_width,
    )
    tile_coordinates: Mapping[TileIdentifier, Mapping[Coordinates, CoordinateValue]] = {
        tile_identifier: {
            Coordinates.X: x_coords,
            Coordinates.Y: y_coords,
            Coordinates.Z: zplane_to_coords[tile_identifier.zplane_label],
        }
        for tile_identifier in tile_identifiers
    }

    write_tile_data(
        tempdir_path,
        FieldOfView.PRIMARY_IMAGES,
        tile_format,
        tile_identifiers,
        tile_fetcher)

    coordinates_csv_path = tempdir_path / "coordinates.csv"
    rows = render_coordinates_to_rows(tile_coordinates)
    write_coordinates_csv(coordinates_csv_path, rows, True)

    # Sleeping by 1 second will result in different timestamps written to TIFF files (the timestamp
    # in the TIFF header has 1 second resolution).  This exposes potential bugs, depending on the
    # nature of the bug and whether in_place is True.
    if tile_format == ImageFormat.TIFF:
        time.sleep(1)

    format_data(
        os.fspath(tempdir_path),
        os.fspath(coordinates_csv_path),
        os.fspath(tempdir_path),
        tile_format,
        in_place,
    )

    # load the data and verify it.
    exp = Experiment.from_json(os.fspath(tempdir_path / "experiment.json"))
    fov = exp.fov()
    stack = fov.get_image(FieldOfView.PRIMARY_IMAGES)

    for round_label in rounds:
        for ch_label in chs:
            for zplane_label in zplanes:
                data, _ = stack.get_slice({
                    Axes.ROUND: round_label, Axes.CH: ch_label, Axes.ZPLANE: zplane_label
                })
                expected_data = unique_data(
                    0, rounds.index(round_label), chs.index(ch_label), zplanes.index(zplane_label),
                    1, len(rounds), len(chs), len(zplanes),
                    tile_height, tile_width,
                )
                assert np.allclose(data, expected_data)

    for selectors in stack._iter_axes({Axes.ZPLANE}):
        zplane_label = selectors[Axes.ZPLANE]
        verify_physical_coordinates(
            stack, x_coords, y_coords, zplane_to_coords[zplane_label], selectors[Axes.ZPLANE])


def format_data(
        image_directory_path: Path,
        coordinates_csv_path: Path,
        output_path: Path,
        tile_format: ImageFormat,
        in_place: bool,
) -> None:
    """Inplace experiment construction monkeypatches the code destructively.  To isolate these side
    effects, we run the experiment construction in a separate process."""
    this = Path(__file__)
    structured_formatter_script_path = this.parent / "structured_formatter_script.py"

    subprocess.check_call(
        [sys.executable,
         os.fspath(structured_formatter_script_path),
         os.fspath(image_directory_path),
         os.fspath(coordinates_csv_path),
         os.fspath(output_path),
         tile_format.name,
         str(in_place),
         ]
    )


class UniqueTiles(FetchedTile):
    """Tiles where the pixel values are unique per fov/round/ch/z."""
    def __init__(
            self,
            # these are the arguments passed in as a result of tile_fetcher_factory's
            # pass_tile_indices parameter.
            fov_id: int, round_id: int, ch_id: int, zplane_id: int,
            # these are the arguments we are passing through tile_fetcher_factory.
            fovs: Sequence[int], rounds: Sequence[int], chs: Sequence[int], zplanes: Sequence[int],
            tile_height: int, tile_width: int,
    ) -> None:
        super().__init__()
        self.fov_id = fov_id
        self.round_id = round_id
        self.ch_id = ch_id
        self.zplane_id = zplane_id
        self.fovs = fovs
        self.rounds = rounds
        self.chs = chs
        self.zplanes = zplanes
        self.tile_height = tile_height
        self.tile_width = tile_width

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: self.tile_height, Axes.X: self.tile_width}

    def tile_data(self) -> np.ndarray:
        """Return the data for a given tile."""
        return unique_data(
            self.fov_id,
            self.rounds.index(self.round_id),
            self.chs.index(self.ch_id),
            self.zplanes.index(self.zplane_id),
            len(self.fovs),
            len(self.rounds),
            len(self.chs),
            len(self.zplanes),
            self.tile_height,
            self.tile_width)


def write_tile_data(
        basepath: Path,
        image_type: str,
        tile_format: ImageFormat,
        tile_identifiers: Sequence[TileIdentifier],
        fetcher: TileFetcher,
        subdir_generator: Optional[Callable[[Path, TileIdentifier], Path]] = None,
) -> None:
    if subdir_generator is None:
        subdir_generator = lambda _basepath, _tile_identifier: _basepath

    for tile_identifier in tile_identifiers:
        fetched_tile = fetcher.get_tile(
            tile_identifier.fov_id,
            tile_identifier.round_label,
            tile_identifier.ch_label,
            tile_identifier.zplane_label)

        subdir = subdir_generator(basepath, tile_identifier)
        subdir.mkdir(parents=True, exist_ok=True)

        tile_path = subdir / (
            f"{image_type}-"
            f"f{tile_identifier.fov_id}-"
            f"r{tile_identifier.round_label}-"
            f"c{tile_identifier.ch_label}-"
            f"z{tile_identifier.zplane_label}."
            f"{tile_format.file_ext}"
        )

        tile_format.writer_func(tile_path, fetched_tile.tile_data())


def render_coordinates_to_rows(
        tile_to_physical_coordinates: Mapping[
            TileIdentifier, Mapping[Coordinates, CoordinateValue]],
) -> Sequence[Mapping[str, str]]:
    results: MutableSequence[Mapping[str, str]] = list()

    for tile_identifier, physical_coordinates in tile_to_physical_coordinates.items():
        rowdata: MutableMapping[str, str] = dict()

        for tile_coordinate_name, tile_coordinate_value in zip(
                TILE_COORDINATE_NAMES, dataclasses.astuple(tile_identifier)
        ):
            rowdata[tile_coordinate_name] = str(tile_coordinate_value)

        for coordinate_name in list(Coordinates):
            coordinate_value = physical_coordinates.get(coordinate_name, None)
            if coordinate_value is None and coordinate_name == Coordinates.Z:
                # Z coordinates may be legitimately missing
                continue

            if isinstance(coordinate_value, tuple):
                rowdata[f'{coordinate_name}_min'] = str(coordinate_value[0])
                rowdata[f'{coordinate_name}_max'] = str(coordinate_value[1])
            else:
                rowdata[f'{coordinate_name}_min'] = str(coordinate_value)

        results.append(rowdata)

    return results


def write_coordinates_csv(
        path: Path,
        rows: Sequence[Mapping[str, str]],
        write_z_coordinates_in_header: bool,
) -> None:
    headers = list(TILE_COORDINATE_NAMES)
    for coordinate_name in list(Coordinates):
        if coordinate_name == Coordinates.Z and not write_z_coordinates_in_header:
            continue
        headers.append(f"{coordinate_name.value}_min")
        headers.append(f"{coordinate_name.value}_max")

    with open(os.fspath(path), "w") as fh:
        writer = csv.DictWriter(fh, headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
