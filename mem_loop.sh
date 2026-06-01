while true; do 
  echo "Starting algorithm at $(date)..."; 
  (ulimit -v 2142880 && timeout --foreground 10h ./build/main -i assets/random-32-32-20.scen -m assets/random-32-32-20.map -N 100 -v 3 -t 20000 -O 2  -max_ll 600 -max_ll_decay 0.6 -wdg -seed 1)
  echo "Process exited or was killed. Restarting in 5 seconds..."; 
  sleep 5; 
done
