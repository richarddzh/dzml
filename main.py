from mmml import helpers


text = "richard 中文"
char_map = helpers.file_read_all_text(r'dataset\ZiDian.txt', 'utf-8')
a = helpers.map_text_to_integers(text, char_map)
print(a)

a = helpers.file_read_text_as_integers(r'dataset\NuoWeiDeSenLin.txt', r'dataset\ZiDian.txt', 'utf-8')
print(a[0:20])
a = helpers.file_read_all_text(r'dataset\NuoWeiDeSenLin.txt', 'utf-8')
print(a[0:20])
