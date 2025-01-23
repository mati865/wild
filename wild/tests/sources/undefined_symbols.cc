
//#AbstractConfig:default
//#DiffEnabled:false
//#RunEnabled:false

//#Config:executable:default
//#LinkArgs:--cc=g++ -fPIC -static

int main() {
  try {
    throw 0;
  } catch (int x) {
    return x;
  }
  return 1;
}
