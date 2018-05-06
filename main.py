import mmml.helpers as helpers
import tensorflow as tf


files = ['BaiYeXing', 'HongGaoLiang', 'TanXiangXing', 'WanLiShiWuNian', 'WenHuaKuLv', 'XingZheWuJiang']
char_map = helpers.file_read_all_text(r'dataset\ZiDian.txt', 'utf-8')

for file in files:
    text = helpers.file_read_all_text('dataset\\' + file + '.txt', 'utf-8')
    new_map = helpers.update_character_map(char_map, text)
    print('{0} to {1}'.format(len(char_map), len(new_map)))
    char_map = new_map

helpers.file_write_all_text(r'dataset\ZiDian.txt', char_map, 'utf-8')