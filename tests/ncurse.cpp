#include <stdlib.h>
#include <ncurses.h>
#include <signal.h>

int main(void) {
  initscr();
  addstr("hellow!");
  move(1,10);
  attron(A_PROTECT);
  addstr("hellow!\n");
  getch();
  refresh();

  endwin();
  return 0;
}

