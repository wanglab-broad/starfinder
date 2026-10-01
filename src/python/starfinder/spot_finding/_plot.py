"""Overlay of one channel's detections on one slice or crop (drawn only when a caller asks; FOV.run never plots)."""
import numpy as np

from starfinder.image import _validate_image


def _channel_index(result, channel, n_channels):
    labels = result.diagnostics.get('channel_labels')
    if isinstance(channel, str):
        if labels is None or channel not in labels:
            raise ValueError(f"channel {channel!r} is not one of the result's channel labels {labels}")
        return list(labels).index(channel)
    if isinstance(channel, (int, np.integer)) and not isinstance(channel, bool) and 0 <= channel < n_channels:
        return int(channel)
    raise ValueError(f"channel must be a channel label or an index below {n_channels}, not {channel!r}")


def plot_detections(image, result, *, channel, z, yx_window=None, round=None, ax=None):
    """Draw the detections of one channel on one Z slice (or a YX crop of it) and return the axes.

    Parameters
    ----------
    image : numpy.ndarray
        The ZYX or ZYXC image the result was detected on (for a result of
        several rounds, the image of the round shown).
    result : SpotFindingResult
        A result with a channel column.
    channel : str or int
        A channel label of the result, or a channel index.
    z : int
        The Z slice; a detection is drawn on the slice its z rounds to
        (half to even).
    yx_window : ((int, int), (int, int)), optional
        Half-open ``((y0, y1), (x0, x1))`` index ranges of the crop; None is
        the whole slice. A detection is drawn when its rounded y and x lie in
        the window.
    round : str, optional
        For a result with a ``round`` column, the round whose detections are
        drawn; None draws every round's.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; None creates a figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axes, with the crop as an image and one marker per detection in
        the window (a single scatter collection).
    """
    import matplotlib.pyplot as plt

    image = _validate_image(image)
    n_channels = image.shape[3] if image.ndim == 4 else 1
    spots = result.spots
    if 'channel' not in spots:
        raise ValueError("plot_detections needs a result with a channel column")
    c = _channel_index(result, channel, n_channels)
    if not isinstance(z, (int, np.integer)) or isinstance(z, bool) or not 0 <= z < image.shape[0]:
        raise ValueError(f"z must be a slice index below {image.shape[0]}, not {z!r}")
    (y0, y1), (x0, x1) = yx_window if yx_window is not None else ((0, image.shape[1]), (0, image.shape[2]))
    if not (0 <= y0 < y1 <= image.shape[1] and 0 <= x0 < x1 <= image.shape[2]):
        raise ValueError(f"yx_window {yx_window!r} is not a nonempty window of the {image.shape[1:3]} slice")
    rows = spots['channel'].to_numpy() == c
    if round is not None:
        if 'round' not in spots:
            raise ValueError("round is given, but the result has no round column")
        rows &= (spots['round'] == round).to_numpy(dtype=bool)
    zyx = spots.loc[rows, ['z', 'y', 'x']].to_numpy()
    index = np.rint(zyx)
    inside = ((index[:, 0] == z) & (index[:, 1] >= y0) & (index[:, 1] < y1)
              & (index[:, 2] >= x0) & (index[:, 2] < x1))
    if ax is None:
        _, ax = plt.subplots()
    plane = image[z, y0:y1, x0:x1, c] if image.ndim == 4 else image[z, y0:y1, x0:x1]
    ax.imshow(plane, cmap='gray', extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5), interpolation='nearest')
    ax.scatter(zyx[inside, 2], zyx[inside, 1], s=40, facecolors='none', edgecolors='tab:red', linewidths=1.0)
    ax.set_xlim(x0 - 0.5, x1 - 0.5)
    ax.set_ylim(y1 - 0.5, y0 - 0.5)
    label = result.diagnostics.get('channel_labels')
    name = label[c] if label is not None else str(c)
    ax.set_title(f"{name}, z={z}" + (f", {round}" if round is not None else "") + f": {int(inside.sum())} detections")
    return ax
