for folder in boolean boolean_2 boolean_3 boolean_4 boolean_5
do
	for problem in AT BW CM DP TA TL
	do 
		cp ~/Desktop/experiments/results/${problem}_2_2/${folder}/* ./${folder}/${problem}
	done
done
