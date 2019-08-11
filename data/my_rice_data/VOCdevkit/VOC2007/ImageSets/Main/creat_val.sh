train_file=val.txt
val_image_path='/home/liufc/data/VOCdevkit/VOC2007/ImageSets/Main/val'
find $val_image_path -name *.jpg > $train_file
sed -i 's/.\{54\}//' $train_file #删除每一行前54个字符，-i实现直接修改文件
sed -i 's/[.jpg]*$//g' $train_file #*$表示末尾，总体表示将以.jpg结尾的.jpg部分替换为空,倒数两条斜杠表示空，g表示替换原来buffer中的，sed在处理字符串的时候并不对源文件进行直接处理，先创建一个buffer，但是加g表示对原buffer进行替换 
