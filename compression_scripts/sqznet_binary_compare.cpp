#include <fstream>
#include <iostream>
#include <stdlib.h>

using namespace std;
enum Layer_type { SQUEEZE3x3, OTHER };
enum Param_type { WEIGHT, BIAS };

int get_wb_stored_num(const int count, Layer_type lt) {
  if (lt == SQUEEZE3x3) {
    return (count - 1) / 3 + 1;
  } else {
    return (count - 1) / 4 + 1;
  }
}

void read_data(ifstream &in, char &min_exp, int &param_count, int &short_count,
               unsigned short *&data, int layer_id, Param_type Pt);
void check_data(int count, const unsigned short *data1,
                const unsigned short *data2, const char *str_wb);
void print_short(const unsigned short *data1, const unsigned short *data2,
                 int count);

int main(int argc, char *argv[]) {
  char *filename1 = argv[1];
  char *filename2 = argv[2];
  ifstream in1(filename1, ios::in | ios::binary);
  if (!in1.is_open()) {
    cout << "Error while opening file!" << filename1 << endl;
    exit(-1);
  }
  ifstream in2(filename1, ios::in | ios::binary);
  if (!in2.is_open()) {
    cout << "Error while opening file!" << filename2 << endl;
    exit(-1);
  }

  for (int i = 0; i < 25; i++) {
    char min_exp_w1, min_exp_b1, min_exp_w2, min_exp_b2;
    int short_count_w1, short_count_b1, short_count_w2, short_count_b2;
    int param_count_w1, param_count_b1, param_count_w2, param_count_b2;
    unsigned short *data_w1 = NULL, *data_b1 = NULL, *data_w2 = NULL,
                   *data_b2 = NULL;

    printf("layer %3d: "
           "============================================================\n",
           i + 1);
    read_data(in1, min_exp_w1, param_count_w1, short_count_w1, data_w1, i,
              WEIGHT);
    read_data(in1, min_exp_b1, param_count_b1, short_count_b1, data_b1, i,
              BIAS);
    read_data(in2, min_exp_w2, param_count_w2, short_count_w2, data_w2, i,
              WEIGHT);
    read_data(in2, min_exp_b2, param_count_b2, short_count_b2, data_b2, i,
              BIAS);

    printf("p_count_w1: %6d, p_count_b1: %6d\n", param_count_w1,param_count_b1);
    printf("p_count_w2: %6d, p_count_b2: %6d\n", param_count_w2,param_count_b2);

    printf("s_count_w1: %6d, s_count_w2: %6d\n", short_count_w1,short_count_b1);
    printf("s_count_b1: %6d, s_count_b2: %6d\n", short_count_w2,short_count_b2);

    printf("min_e_w1: %3d, min_e_b1: %6d\n", min_exp_w1, min_exp_b1);
    printf("min_e_w2: %3d, min_e_b2: %6d\n", min_exp_w2, min_exp_b2);

    if (param_count_w1 != param_count_w2 || param_count_b1 != param_count_b2 ||
        min_exp_w1 != min_exp_w2 || min_exp_b1 != min_exp_b2) {
      printf("Error: param_count different!\n");
      in1.close();
      in2.close();
      exit(-1);
    }
    // check the short weight

    if (i == 1) {
      print_short(data_w1, data_w2, short_count_w1);
      print_short(data_b1, data_b2, short_count_b1);
    }
    check_data(short_count_w1, data_w1, data_w2, "weight");
    check_data(short_count_b1, data_b1, data_b2, "bias");
    free(data_w1);
    data_w1 = NULL;
    free(data_w2);
    data_w2 = NULL;
    free(data_b1);
    data_b1 = NULL;
    free(data_b2);
    data_b2 = NULL;
  }
  in1.close();
  in2.close();
}

void read_data(ifstream &in, char &min_exp, int &param_count, int &short_count,
               unsigned short *&data, int layer_id, Param_type Pt) {

  cout << "current file pos: " << in.tellg() << endl;
  // printf("current file pos: %d\n", in.tellg());
  in.read(&min_exp, sizeof(char));
  in.read(reinterpret_cast<char *>(&param_count), sizeof(int) / sizeof(char));
  if ((layer_id % 3) == 0 && Pt == WEIGHT) {
    short_count = get_wb_stored_num(param_count, SQUEEZE3x3);
  } else {
    short_count = get_wb_stored_num(param_count, OTHER);
  }

  data = (unsigned short *)malloc(short_count * sizeof(unsigned short));
  in.read(reinterpret_cast<char *>(data),
          short_count * sizeof(unsigned short) / sizeof(char));
}

void check_data(int count, const unsigned short *data1,
                const unsigned short *data2, const char *str_wb) {
  int num_diff = 0;
  printf("%s diff:----------\n", str_wb);
  for (int j = 0; j < count; j++) {
    unsigned short tmp1 = data1[j];
    unsigned short tmp2 = data2[j];
    if (tmp1 != tmp2) {
      num_diff++;
      printf("%s_short_id[%d] %x != %x\n", str_wb, j, tmp1, tmp2);
    }
  }
  printf("%d / %d [%f] different!\n", num_diff, count, float(num_diff) / count);
  if (num_diff != 0) {
    printf("!!!!!!!!!!!!!!!!!!\n\n");
  }
}

void print_short(const unsigned short *data1, const unsigned short *data2,
                 int count) {
  printf("--------------------short compare 1----------------------\n");
  for (int i = 0; i < count; i++) {
    printf("%04x ", data1[i]);
    if ((i + 1) % 10 == 0) {
      printf("\n");
    }
  }
  printf("\n");
  printf("--------------------short compare 2----------------------\n");
  for (int i = 0; i < count; i++) {
    printf("%04x ", data2[i]);
    if ((i + 1) % 10 == 0) {
      printf("\n");
    }
  }
  printf("\n");
}
