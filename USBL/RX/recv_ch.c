#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

int main(){
	//usleep(10000000);
	system("/usr/local/bin/sdmsh/sdmsh -f /usr/local/bin/sdmsh/rx_ch.script 192.168.42.2");
	usleep(1000000);
	return 0;
}
