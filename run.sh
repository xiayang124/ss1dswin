

for i in 5 10 15 20 25 30 0.1
do
      echo "runing Indian"
      python demo.py --data="Indian" --train_num=i --epoch=3000 --seed=114
done

for i in 5 10 15 20 25 30 0.1
do
      echo "runing Honghu"
      python demo.py --data="Honghu" --train_num=i --epoch=3000 --seed=114
done

for i in 5 10 15 20 25 30 0.1
do
      echo "runing Pavia"
      python demo.py --data="Pavia" --train_num=i --epoch=3000 --seed=114
done