#include "EasyNextionLibrary.h";  // Library for interfacing with Nextion touchscreens

EasyNex myNex(Serial1);           // Small screen (e.g., "Start" button)
EasyNex myNex2(Serial2);          // Large screen (e.g., for red/green choice)

// LiquidCrystal lcd(12, 11, 5, 4, 3, 2);
// Peripherals
const int SpeakerPin = 8;         // Piezo speaker for auditory feedback
const int RedLEDPin = 22;         // Red LED (Choice A)
const int GreenLEDPin = 23;       // Green LED (Choice B)
const int PumpPin = 4;            // Reward pump control
const int SENSORPIN = 6;          // IR sensor for reach detection (currently unused)
const int RewardSigPin = 24;      // Output to signal DAQ that a reward was delivered
const int SmallScrPin = 25;       // DAQ signal pin for small screen press
const int LargeScrPin = 27;       // DAQ signal pin for large screen press

// int PumpMillis = 0; //variable for reward pump on/off
int LightSequence[100];           // Which LED was shown per trial
int Choice[100];                  // What the subject chose per trial
int Trial = 0;                    // Trial index
int NumCorrect = 0;               // Correct responses counter
int NumIncorrect = 0;             // Incorrect responses counter

enum class GameState : uint8_t {
  STATE_IDLE,       // Waiting for start button press
  STATE_CHOICE,     // Waiting for subject's response
  STATE_VERIFY,     // Evaluate correctness
  STATE_CORRECT,    // Correct response handling
  STATE_INCORRECT   // Incorrect response handling
};

GameState gameState;

//Declaring function to start game
void StartGame(){
  myNex2.writeStr("sleep=1");     // Turn off large screen (sleep mode)
  myNex.NextionListen();          // Wait for a touch event from small screen (Serial1)
}

void Reward(){
  digitalWrite(PumpPin, HIGH);           // Turn on the pump
  digitalWrite(RewardSigPin, HIGH);      // Signal DAQ that reward started
  delay(4);                              // Short high pulse for DAQ
  digitalWrite(RewardSigPin, LOW);       // Return DAQ line low
  delay(1000);                           // Pump duration (1s)
  digitalWrite(PumpPin, LOW);            // Turn off pump
  Serial.println("Delivering Reward");
}

// Start button pressed
void trigger1(){
  digitalWrite(SmallScrPin, HIGH);  // Signal small screen press to DAQ
  delay(4);
  digitalWrite(SmallScrPin, LOW);

  Serial1.end();                    // Disable further input from small screen

  myNex2.writeStr("sleep=0");       // Wake up large screen
  delay(50);                        // Wait for screen to turn on

  int RandPage = random(0,8);       // Random page for touchscreen (0–7)
  myNex2.writeStr("page " + String(RandPage));
  Serial.println("Trial: " + String(Trial + 1));

  tone(SpeakerPin, 262, 1000);      // Play start tone

  int RandLED = random(22,24);      // Randomly pick between Red (22) and Green (23)
  digitalWrite(RandLED, HIGH);      // Turn on LED
  LightSequence[Trial] = RandLED;   // Log which LED was shown
  if(digitalRead(GreenLEDPin) == HIGH){
    myNex.writeNum("b0.bco", 2016); // Set button b0 background color to GREEN (color code: 2016)
    myNex.writeStr("b0.txt", "GREEN"); // Set button b0 text to "ON"
  }else if(digitalRead(RedLEDPin) == HIGH){ 
    myNex.writeNum("b0.bco", 63488); // Set button b0 background color to RED (color code: 63488)
    myNex.writeStr("b0.txt", "RED"); // Set button b0 text to "ON"
  }
  gameState = GameState::STATE_CHOICE;
}



//Declaring function for recording the choice of the player
void PlayerChoice(){
  myNex2.NextionListen();           // Waits for large screen interaction (touch input)
}

// Red Button presssed
void trigger2(){
  digitalWrite(LargeScrPin, HIGH); delay(4); digitalWrite(LargeScrPin, LOW);
  myNex2.writeStr("sleep=1"); delay(30);
  Serial.println("Red button pressed!");
  Choice[Trial] = RedLEDPin;
  tone(SpeakerPin, 415, 1000);
  gameState = GameState::STATE_VERIFY;
}

void trigger3(){
  digitalWrite(LargeScrPin, HIGH); delay(4); digitalWrite(LargeScrPin, LOW);
  myNex2.writeStr("sleep=1"); delay(30);
  Serial.println("Green button pressed!");
  Choice[Trial] = GreenLEDPin;
  tone(SpeakerPin, 370, 900);
  gameState = GameState::STATE_VERIFY;
}


//declare function to verify choice as correct/incorrect
void VerifyChoice(){
  if(LightSequence[Trial] == Choice[Trial]){
    Serial.println("Correct!");
    gameState = GameState::STATE_CORRECT;
  }else{
    Serial.println("Incorrect...");
    gameState = GameState::STATE_INCORRECT;
  }
}

//declare function when choice is correct
void CorrectChoice(){
  Reward();
  digitalWrite(RedLEDPin,HIGH);
  digitalWrite(GreenLEDPin,HIGH);
  delay(1000);
  digitalWrite(RedLEDPin,LOW);
  digitalWrite(GreenLEDPin,LOW);
  Trial = Trial + 1;
  NumCorrect = NumCorrect + 1;  
  Serial1.begin(9600);   // Re-enable small screen for next trial
  delay(50);             // Give it time to sync
  gameState = GameState::STATE_IDLE;
}

//declare function when choice is incorrect
void IncorrectChoice(){
  Reward();
  Trial = Trial + 1;
  NumIncorrect = NumIncorrect + 1;    
  Serial1.begin(9600);   // Re-enable small screen for next trial
  delay(50);             // Give it time to sync
  gameState = GameState::STATE_IDLE;
}

void setup() {
  Serial.begin(9600);
  myNex.begin(9600);
  myNex2.begin(9600);
  pinMode(RedLEDPin,OUTPUT);
  pinMode(GreenLEDPin,OUTPUT);
  pinMode(PumpPin,OUTPUT);
  pinMode(SENSORPIN, INPUT);
  pinMode(RewardSigPin, OUTPUT);
  pinMode(SmallScrPin, OUTPUT);
  pinMode(LargeScrPin, OUTPUT);
  digitalWrite(SENSORPIN, HIGH); 
  gameState = GameState::STATE_IDLE;
}

void loop() {
  switch(gameState){
    case GameState::STATE_IDLE: //waiting for button push
      StartGame();
      break;
  
    case GameState::STATE_CHOICE:  //Player must choose RedButton or GreenButton, store in Choice array
      PlayerChoice();
      // Serial.println(digitalRead(SENSORPIN));
      // //For X trials, if IR beam is broken, then change page on big screen
      // int sensorState = 0;
      // sensorState = digitalRead(SENSORPIN);
      // if (Trial % 2){ 
      //   if (sensorState == 0){
      //   myNex2.writeStr("page 2");
      //   //Serial.println("Broken");
      //   }
      // }
      break;
    case GameState::STATE_VERIFY:  //Check index of Choice versus index of LightSequence
      Serial.println("Checking guess");
      VerifyChoice();
      break;

    case GameState::STATE_CORRECT: //Player choice is correct, record # correct on LCD, trigger reward via Rexx
      Serial.println("Correct");
      CorrectChoice();
      break;
      
    case GameState::STATE_INCORRECT: //Player choice is incorrect, record # incorrect on LCD, no reward
      Serial.println("Incorrect");
      IncorrectChoice();
      break;
  }
}
