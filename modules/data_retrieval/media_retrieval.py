from PIL import Image
from PIL import UnidentifiedImageError
import requests
import re
from requests.exceptions import RequestException
from modules.log import logger


def download_image(url):
    if url is None or isinstance(url, float):
        return url
    url = url.replace('"', '')
    try:
        url = requests.get(url, stream=True).raw
    except RequestException:
        pass
    try:
        e = ""
        k = 10
        while k > 0:
            try:
                return Image.open(url).convert("RGB")
            except UnidentifiedImageError as err:
                k -= 1
                e = err
        assert k <= 0
    except AssertionError:
        logger.warning(f"Cannot retrieve image from url:\n{url}\n{e}")
        return None


def filter_video_list(lst):
    if type(lst[0]) is not str:
        return lst
    r = re.compile("^.+[0-9]+_[0-9]+_[0-9]+k\\..+$")
    return list(filter(lambda x: not r.match(x), lst))
