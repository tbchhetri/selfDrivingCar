const int EnableL = 5;
const int HighL = 6;
const int LowL = 7;

void setup() {
  pinMode(EnableL, OUTPUT);
  pinMode(HighL, OUTPUT);
  pinMode(LowL, OUTPUT);
}

void forward() {
  digitalWrite(HighL, LOW);
  digitalWrite(LowL, HIGH);
  analogWrite(EnableL, 255);
}

void loop (){
forward();
}
