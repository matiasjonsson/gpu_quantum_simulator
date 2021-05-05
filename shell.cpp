#include "shell.hpp"




int main(){
  //char command[1024]="";
  string command;
  while(true){
    std::cin >> command;
    std::cout << command << std::endl;

    if(command.compare(string("hello")) == 0 ) break;
  }
  return 0;
}
