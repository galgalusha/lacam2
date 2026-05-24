while true; do 
  echo "Starting algorithm at $(date)..."; 
  (ulimit -v 6242880 && timeout --foreground 2h ./build/main -i assets/random-32-32-20.scen -m assets/random-32-32-20.map -N 100 -v 3 -t 120000 -O 2 -seed 2209149 -wdg); 
  echo "Process exited or was killed. Restarting in 5 seconds..."; 
  sleep 5; 
done
