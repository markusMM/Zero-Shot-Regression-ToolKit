import os
import numpy as np
import pytest

from modules.data_retrieval.media_retrieval import download_image, filter_video_list


class TestDownloadImage:

    im = os.path.join(os.getcwd(), "modules_test/test_data/images/image.png")

    @pytest.mark.dependency(name="dlim:size")
    def test_image_dims(self):
        im = download_image(self.im)
        assert im.size == (600, 401)

    @pytest.mark.dependency(name="dlim:quote")
    def test_quotes(self):
        im = download_image('"' + self.im + '"')
        assert im.size == (600, 401)

    @pytest.mark.dependency(name="dlim:none")
    def test_missing(self):
        im = download_image(.98)
        assert im == .98
        im = download_image(None)
        assert im is None
        im = download_image(np.nan)
        assert im != im


class TestFilterVideoList:

    video_list = ["video.mp4", "video_756k_800_1200_756k.mp4"]

    @pytest.mark.dependency(name="filvid:length")
    def test_length(self):
        assert len(filter_video_list(self.video_list)) == 1

    @pytest.mark.dependency(name="filvid:values")
    def test_values(self):
        assert filter_video_list(self.video_list) == ["video.mp4"]
