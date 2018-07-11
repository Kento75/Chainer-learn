# -*- coding: utf-8 -*-
import codecs
import re
import urllib.parse
import urllib.request
import os
import socket
from PIL import Image

# 保存場所の作成
if not os.path.isdir('portrait'):
    os.mkdir('portrait')
if not os.path.isdir('train'):
    os.mkdir('train')

# URLのリスト
base_url = 'https://commons.wikimedia.org'
url = base_url + '/wiki/Category:17th-century_oil_portraits_of_standing_women_at_three-quarter_length'
suburl = base_url + '/wiki/File:'
next_page = url

# タイムアウトを設定
socket.setdefaulttimeout(10)
# 画像サイズの上限を廃止
Image.MAX_IMAGE_PIXELS = None

# スクレイピング
while len(next_page) > 0:
    url = next_page
    next_page = ''
    # 日本語版Wikimediaのページ
    with urllib.request.urlopen(url) as response:
        # URLから読み込む
        html = response.read().decode('utf-8')

        # ページタイトルと次のページへのリンクを取得
        title = re.findall(r'<title>([\s\S]*) - Wikimedia Commons</title>', html)
        if len(title) < 1:
            break
        nextpage = re.findall( \
            r'<a\s*href=\"(/w/index.php?[\s\S]*)\" title=\"' + title[0] + '\">[\s\S]*>next page</a>', \
            html)

        # ギャラリー表示部分のタグを取得する
        gallery = re.findall( \
            r'<div class=\"gallerytext\">\s+<a\s+href=\"/wiki/File:(\S*)\"', \
            html, re.DOTALL)

        # ギャラリーを開く
        for g in gallery:
            # サブページを開く
            with urllib.request.urlopen(suburl + g) as response:
                g = urllib.parse.quote_plus(urllib.parse.unquote_plus(g))
                # URLから読み込む
                html = response.read().decode('utf-8')
                original = re.findall( \
                    r'<a\s+(?:class=\"internal\")?\s*href=\"(https://upload.wikimedia.org/\S*/' + g + ')\"[\s\S]*>[\s\S]*</a>', \
                    html)
                # 画像をダウンロード
                for o in original:
                    face = o.rsplit('/', 1)[1]
                    os.system('wget ' + o + ' -O portrait/' + face)
                    # TIFファイルはJpegに変換する
                    if face.endswith('.tif') or face.endswith('.tiff'):
                        os.system(
                            'gdal_translate -of JPEG -ot Byte -co QUALITY=100 portrait/' + face + ' portrait/' + face + '.jpg')
                        os.remove('portrait/' + face)
                        os.remove('portrait/' + face + '.jpg.aux.xml')

        # 次のページのURLを作る
        if len(nextpage) > 0:
            next = nextpage[0].replace('&amp;', '&')
            next_page = base_url + next
        else:
            next_page = ''

# 320x320のデータセットを作る
fs = os.listdir('portrait')
numimg = 0
for fn in fs:
    # 画像を読み込み
    img = Image.open('portrait/' + fn).convert('RGB')
    # 上部中央の顔付近を切りだす
    w = img.size[0] / 2
    x = img.size[0] / 4
    img = img.crop((x, 0, x + w, w)).resize((320, 320))
    # 名前を付けて保存する
    img.save('train/' + str(numimg) + '.png')
    numimg = numimg + 1
