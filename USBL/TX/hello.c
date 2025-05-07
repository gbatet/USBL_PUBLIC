#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

int main(){
   usleep(10000000);
   while(1){
        system("nice -n -20 /usr/local/bin/sdmsh/sdmsh -f /usr/local/bin/sdmsh/tx.script 192.168.42.2"); 
      	usleep(10000000);
    }
    return 0;
}
