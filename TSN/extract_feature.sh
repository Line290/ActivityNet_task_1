for i in `seq 0 15`
do
   CUDA_VISIBLE_DEVICES=$((i%4)) python extract_anet_feature.py ./lists/list.txt.new$i ./dpn92_rgb_model_best.pth.tar -m RGB --arch dpn92 -s nonrescale &
done

for i in `seq 0 15`
do
   CUDA_VISIBLE_DEVICES=$((i%4)) python extract_anet_feature.py ./lists/list.txt.new$i ./dpn92_flow__flow_model_best.pth.tar -m Flow --arch dpn92 -s nonrescale &
done
