#include <LiquidCrystal_I2C.h>
#include <Stepper.h>
#include <Wire.h>
#include "Adafruit_VL6180X.h"

// ============================================================
// ERROR CODES — mirrors error_codes.py on the PC side
// Format: <Error,EXXX,short description>
// ============================================================
#define EC_TOF_NOT_FOUND      "E501"   // ToF sensor not found on I2C
#define EC_TOF_SYS_ERR        "E502"   // ToF system error
#define EC_TOF_ECE            "E503"   // ToF ECE failure
#define EC_TOF_NO_CONVERGE    "E504"   // ToF no convergence
#define EC_PICKUP_RETRY       "E505"   // Pickup retry limit reached
#define EC_OVERFLOW_FULL      "E506"   // Overflow tray full
#define EC_CMD_NOT_STARTED    "E507"   // Tray command rejected: machine stopped
#define EC_CMD_STARTED        "E508"   // Manual command rejected: machine running
#define EC_NOT_AT_HOME        "E509"   // Parameter change rejected: not at home
#define EC_HOME_X_FAIL        "E510"   // Homing failed: X endstop
#define EC_HOME_Y_FAIL        "E511"   // Homing failed: Y endstop
#define EC_HOME_Z_FAIL        "E512"   // Homing failed: Z endstop
#define EC_BUF_OVERFLOW        "E513"   // Serial buffer overflow / truncation
#define EC_UNKNOWN_CMD         "E514"   // Unrecognised command received

// Helper macro: sends a structured error over serial and shows code on LCD line 2
// Usage: REPORT_ERROR(EC_TOF_NOT_FOUND, "No ToF sensor")
#define REPORT_ERROR(code, msg) \
  do { \
    Serial.println("<Error," + String(code) + "," + String(msg) + ">"); \
    lcd.setCursor(0, 1); lcd.print(String(code) + " " + String(msg).substring(0, 11)); \
  } while(0)

//Pin Definitions for hardware control
enum Pins { 
    Zmin = 18, Xmin = 3, Ymin = 14, Xmax = 2, Ymax = 15, Zmax = 19, //Endstops
    Lights = 8, Vacuum1 = 9, Vacuum2 = 10, //Vacuums: Pickup/Release respectively
    Xenable = 38, Yenable = 56, Zenable = 62, E0enable = 24, E1enable = 30, //Motors
    Xstep = 54, Ystep = 60, Zstep = 46, E0step = 26, E1step = 36, //Stepper Controls
    Xdir = 55, Ydir = 61, Zdir = 48, E0dir = 28, E1dir = 34 //Directions
};

//Initialize sensor and display objects
Adafruit_VL6180X vl = Adafruit_VL6180X();  //VL6180X distance sensor object
LiquidCrystal_I2C lcd(0x27, 16, 2);        //LCD for user feedback

//Flags and control variables
boolean newData = false, readInProgress = false, newDataFromPC = false;
boolean atHomePosition = true;  //Track if machine is at home position (safe for parameter changes)
boolean machineStarted = false; //Track if machine is in Started state (accepting tray commands)
byte bytesRecvd = 0, PickupRetry;
const byte numChars = 64, buffSize = 40;                                               //Buffer sizes for serial communication
const char startMarker = '<', endMarker = '>';                                         //Start/End markers for data communication
char inputBuffer[buffSize], messageFromPC[buffSize] = { 0 }, receivedChars[numChars];  //Buffers for data

//Timers and counters
float timeoutcount = 0;
const float timeout = 10;

//Coordinates and distance tracking arrays
const char* MatchingValues[] = {"RejectCard", "tray1", "tray7", "tray14", "tray18","tray25", "tray26", "tray27", "tray28", "tray29", "tray30", "tray31", "tray32"};
const int loopStart[] = {33, 1, 7, 14, 18, 25, 26, 27, 28, 29, 30, 31, 32};
const int loopEnd[] = {33, 6, 13, 17, 24, 25, 26, 27, 28, 29, 30, 31, 32};
boolean match = 0;
const int xOffsets[6] = {-3, -2, -1, 1, 2, 3}; const int yOffsets[4] = {-2, -1, 1, 2}; //Arrays for X/Y coordinate movements
const short X[6][5]={{31,27,21,28,32},{23,11,9,12,24},{17,5,1,6,18},{19,7,2,8,20},{25,13,10,14,26},{33,29,22,30,34}}; //Array for X trays
const short Y[4][7]={{32,24,18,16,20,26,34},{28,12,6,4,8,14,30},{27,11,5,3,7,13,29},{31,23,17,15,19,25,33}}; //Array for Y trays
short CountArray[35], upcount; //Array is for keeping track of how many cards have been moved to each tray
String AssignedTrayValue[35], Tempval1; //Array for keeping try of tray assignments
uint8_t range[6]; //Range readings from VL6180X sensor

//Endstop tracking variables (limits of movement for each axis)
byte X_ENDSTOP_MIN, Y_ENDSTOP_MIN, Z_ENDSTOP_MIN, X_ENDSTOP_MAX, Y_ENDSTOP_MAX, Z_ENDSTOP_MAX;

//Movement parameters (calibration and speed settings)
short initial_pickup_distance = 6000, initial_drop_distance = 4000;  //Intial movement distances
short Xcal = 350, Ycal = 475, Zcal = 140;                            //Movement multipliers
short speed = 700, zspeed = 75, zespeed = 120;                       //Movement speed (Higher number=slower speed)
short pickup_threshold = 40, release_threshold = 40;                 //Thresholds for pickup/release conditions
short HCC = 10, YCourseCorrection = 1, XCourseCorrection = 0;        //Cycles until rehome and course correction variables

void setup() {
  byte pins[] = { Xstep, Ystep, Zstep, E0step, E1step, Xdir, Ydir, Zdir, E0dir, E1dir, Xenable, Yenable, Zenable, E0enable, E1enable, Vacuum1, Vacuum2, Lights};
  for (byte pin : pins) { pinMode(pin, OUTPUT); }  //Set all pins as outputs
  
  // Motors enabled (LOW), vacuums off (LOW), lights ON (HIGH) by default
  digitalWrite(Xenable, LOW);    //X motor enabled
  digitalWrite(Yenable, LOW);    //Y motor enabled  
  digitalWrite(Zenable, LOW);    //Z motor enabled
  digitalWrite(E0enable, LOW);   //E0 motor enabled
  digitalWrite(E1enable, LOW);   //E1 motor enabled
  digitalWrite(Vacuum1, LOW);    //Vacuum1 off
  digitalWrite(Vacuum2, LOW);    //Vacuum2 off
  digitalWrite(Lights, HIGH);    //Lights on by default

  Serial.begin(9600);                        //Begin serial connection
  digitalWrite(13, HIGH);                    //Turn on the onboard LED for debugging
  while (!Serial) { delay(10); }             //Wait for serial to be ready
  Serial.println("Adafruit VL6180x test!");  //Print sensor initialization message
  while (!vl.begin()) {
    REPORT_ERROR(EC_TOF_NOT_FOUND, "No ToF sensor");
    PrintLCD("E501 No ToF", "Check I2C wiring");
    delay(1000);
  }  
  Serial.println("Sensor found!");PrintLCD("ToF sensor", "found");
  Homemachine(); //Call Homemachine to calibrate the system
  delay(100);Serial.println("<Arduino is ready>");
}

void loop() {
  uint8_t status = vl.readRangeStatus();
  if (AssignedTrayValue[34] != "OverflowTray") {AssignedTrayValue[34] = "OverflowTray";} //Assign some initial tray values that are reserved
  if (AssignedTrayValue[33] != "RejectCard") {AssignedTrayValue[33] = "RejectCard";}
  getDataFromPC(); //Retrieve data from PC

  //Check sensor error status and handle errors accordingly
  if (status == VL6180X_ERROR_NONE) {ReadRange(1); Serial.print("Range: "); Serial.println((range[0] + range[1]) / 2);
  } else {PrintLCD("ToF: No range", " ");}
  switch (status) { //Various errors that can happen with the range sensor
    case VL6180X_ERROR_SYSERR_1 ... VL6180X_ERROR_SYSERR_5:
      REPORT_ERROR(EC_TOF_SYS_ERR, "System error"); break;
    case VL6180X_ERROR_ECEFAIL:
      REPORT_ERROR(EC_TOF_ECE, "ECE failure"); break;
    case VL6180X_ERROR_NOCONVERGE:
      REPORT_ERROR(EC_TOF_NO_CONVERGE, "No convergence"); break;
    case VL6180X_ERROR_RANGEIGNORE:
      Serial.println("<Info,E504,Ignoring range>"); break;
    case VL6180X_ERROR_SNR:
      REPORT_ERROR(EC_TOF_SYS_ERR, "Signal/Noise err"); break;
    case VL6180X_ERROR_RAWUFLOW:
      REPORT_ERROR(EC_TOF_SYS_ERR, "Raw underflow"); break;
    case VL6180X_ERROR_RAWOFLOW:
      REPORT_ERROR(EC_TOF_SYS_ERR, "Raw overflow"); break;
    case VL6180X_ERROR_RANGEUFLOW:
      REPORT_ERROR(EC_TOF_SYS_ERR, "Range underflow"); break;
    case VL6180X_ERROR_RANGEOFLOW:
      REPORT_ERROR(EC_TOF_SYS_ERR, "Range overflow"); break;
    default: break;
  }
  lcd.init();lcd.backlight();lcd.setCursor(0, 0); //Initialize LCD and turn on the backlight
  ReadEndstops(); //Read endstop values 
  //if(Y_ENDSTOP_MIN==0||Z_ENDSTOP_MIN==0||X_ENDSTOP_MAX==0){PrintLCD("Endstop reached"," ");STOP==1;delay(100);}
  PrintLCD("Ready", " "); //Display "Ready" message on LCD 
  Tempval1 = Serial.readString();delay(10); //read any available data from the serial buffer
  if (Tempval1 != "") {
    PrintLCD("Received: ", Tempval1); //Display the received data on the LCD
    ReadRange(1);DetermineAction(); //Decide what action to take based on received data
    Serial.println("<Arduino is ready>");
    timeoutcount = 0; //Reset timeout counter
  }  
  //timeoutcount++;if(timeoutcount>=timeout){Tray(34);Serial.println("<Arduino is ready>");timeoutcount=0;}
  //Serial.println(timeoutcount);

  //Reset flags
  Tempval1 = "";messageFromPC == "";bytesRecvd == 0;newData = false;readInProgress = false;newDataFromPC = false;
}  

void getDataFromPC() {
  if (Serial.available() > 0) {
    char x = Serial.read();  //Read incoming character
    if (x == endMarker) { //End marker received
      readInProgress = false;
      newDataFromPC = true;
      inputBuffer[bytesRecvd] = 0; //null terminate input buffer 
      parseData(); //parse the received data
    }   
    if (readInProgress) {
      inputBuffer[bytesRecvd] = x;
      bytesRecvd++;  //Data is still being read
      if (bytesRecvd == buffSize) {
        bytesRecvd = buffSize - 1; //Prevent buffer overflow
        REPORT_ERROR(EC_BUF_OVERFLOW, "Cmd truncated");
      }
    }  
    if (x == startMarker) { //Start marker received, begin reading
      bytesRecvd = 0;
      readInProgress = true;
    }
  }
}  

//Parse the received data from the buffer
void parseData() {
  char* strtokIndx;
  strtokIndx = strtok(inputBuffer, ",");  //Split the string by commas
  strcpy(messageFromPC, strtokIndx);      //Store the first token
  strtokIndx = strtok(NULL, ",");
  strtokIndx = strtok(NULL, ","); //Continue parsing if necessary
}

//Handle the picking process, with retries
void pick(short steps, byte Release) {PickupRetry = 0;
retrypickup:
  Move1(0, steps, zspeed);  //Move to pickup position
  if (Release == 1) {MotorsOnOff(1);delay(100);digitalWrite(Vacuum2,1);delay(300);digitalWrite(Vacuum2,0);ReadRange(3);  //Turn motors off and release card
    while ((range[2] + range[3]) / 2 < release_threshold) {digitalWrite(Vacuum2,1);delay(300);digitalWrite(Vacuum2,0);ReadRange(3);}//Check range and if card didn't drop retry
    MotorsOnOff(0);delay(100);Move1(1, steps, zespeed);} //Turn motors back on and move back up
  if (Release == 0) {digitalWrite(Vacuum1, 1);delay(500);digitalWrite(Vacuum1, 0);Move1(1, steps, zespeed);}ReadRange(3);//Vacuum to pick up card
    if (((range[2] + range[3]) / 2) > pickup_threshold) {
      PickupRetry++;
      if (PickupRetry >= 10) {
        REPORT_ERROR(EC_PICKUP_RETRY, "10 retries failed");
        StopMachine();
      } else { goto retrypickup; }
    }//If 10 retrys failed stop the machine
}  

//Homing routine to calibrate the machine
void Homemachine() {PrintLCD("Calibrating", " ");ReadEndstops();
  // Z axis home
  if (Z_ENDSTOP_MIN == 1) {
    Move1(0, 800, zespeed);
    unsigned long t0 = millis();
    while (Z_ENDSTOP_MIN == 1) {
      Move1(1, 3, zespeed); Z_ENDSTOP_MIN = digitalRead(Zmin);
      if (millis() - t0 > 10000) { REPORT_ERROR(EC_HOME_Z_FAIL, "Z endstop"); break; }
    }
    delay(200);
  }
  // Y axis home
  if (Y_ENDSTOP_MIN == 1) {
    Move4(1, 0, 0, 50);
    unsigned long t1 = millis();
    while (Y_ENDSTOP_MIN == 1) {
      Move4(0, 1, 0, 5); Y_ENDSTOP_MIN = digitalRead(Ymin);
      if (millis() - t1 > 10000) { REPORT_ERROR(EC_HOME_Y_FAIL, "Y endstop"); break; }
    }
    delay(200);
  }
  // X axis home
  if (X_ENDSTOP_MAX == 1) {
    Move4(1, 0, 50, 0);
    unsigned long t2 = millis();
    while (X_ENDSTOP_MAX == 1) {
      Move4(0, 1, 3, 0); X_ENDSTOP_MAX = digitalRead(Xmax);
      if (millis() - t2 > 10000) { REPORT_ERROR(EC_HOME_X_FAIL, "X endstop"); break; }
    }
    delay(200);
  }
  Move1(0, 3000, zspeed); delay(200); Move4(1, 0, Xcal * 3 + 55, Ycal * 2 + 38); delay(200);
}  

void StopMachine() {
  REPORT_ERROR(EC_PICKUP_RETRY, "Machine halted");
  PrintLCD("E505 Halted", "Empty and reset"); MotorsOnOff(0); while (1) { delay(10000);}
}

void ReadEndstops() { //Read the endstop sensors
  X_ENDSTOP_MIN = digitalRead(Xmin);X_ENDSTOP_MAX = digitalRead(Xmax);
  Y_ENDSTOP_MIN = digitalRead(Ymin);Y_ENDSTOP_MAX = digitalRead(Ymax);
  Z_ENDSTOP_MIN = digitalRead(Zmin);Z_ENDSTOP_MAX = digitalRead(Zmax);
}

void MotorsOnOff(boolean OnOff) { //Turns all motors on or off
  digitalWrite(Xenable, OnOff);digitalWrite(E0enable, OnOff);
  digitalWrite(Yenable, OnOff);digitalWrite(E1enable, OnOff);
  digitalWrite(Zenable, OnOff);
}

void DetermineAction() { //Figure out what to do
  // Parse optional step count from manual movement commands (e.g. "CalibrateX1,10")
  int manualSteps = 5; // default steps per button press
  int commaIdx = Tempval1.indexOf(',');
  if (commaIdx > 0) {
    String stepStr = Tempval1.substring(commaIdx + 1);
    stepStr.trim();
    int parsed = stepStr.toInt();
    if (parsed > 0) manualSteps = parsed;
    Tempval1 = Tempval1.substring(0, commaIdx); // strip suffix so equality checks below still work
  }

  // Start/Stop commands
  if (Tempval1 == "StartMachine") {
    machineStarted = true;
    Serial.println("<OK,MachineStarted>");
    PrintLCD("Machine STARTED", "Accepting cards");
    return;
  } else if (Tempval1 == "StopMachine") {
    machineStarted = false;
    Serial.println("<OK,MachineStopped>");
    PrintLCD("Machine STOPPED", "Controls enabled");
    return;
  }
  
  // Tray commands only allowed when machine is started
  for (int i = 0; i < sizeof(MatchingValues) / sizeof(MatchingValues[0]); ++i) {
    if (Tempval1 == MatchingValues[i]) {
      if (!machineStarted) {
        REPORT_ERROR(EC_CMD_NOT_STARTED, "Start machine");
        PrintLCD("E507 Stopped", "Start machine!");
        return;
      }
      match = 1; atHomePosition = false; ForLoop(loopStart[i], loopEnd[i]); atHomePosition = true; 
      break;
    }
  }
  
  // Manual controls only allowed when machine is stopped
  if (Tempval1 == "CalibrateX1" || Tempval1 == "CalibrateX2" ||
      Tempval1 == "CalibrateY1" || Tempval1 == "CalibrateY2" ||
      Tempval1 == "HomeButton") {
    if (machineStarted) {
      REPORT_ERROR(EC_CMD_STARTED, "Stop machine");
      PrintLCD("E508 Started", "Stop machine!");
      return;
    }
  }
  
  // Calibration commands (minor movements)
  if (Tempval1 == "CalibrateX1") {Move4(0,1,manualSteps,0);
  } else if (Tempval1 == "CalibrateX2") {Move4(1,1,manualSteps,0);
  } else if (Tempval1 == "CalibrateY1") {Move4(0,0,0,manualSteps);
  } else if (Tempval1 == "CalibrateY2") {Move4(0,1,0,manualSteps);
  } else if (Tempval1 == "HomeButton") {Homemachine();atHomePosition = true;
  
  // Query sensor data
  } else if (Tempval1 == "QuerySensors") {
    ReadRange(1);
    ReadEndstops();
    String response = "<Sensors,range=";
    response += String((range[0] + range[1]) / 2);
    response += ",xmin=" + String(X_ENDSTOP_MIN);
    response += ",xmax=" + String(X_ENDSTOP_MAX);
    response += ",ymin=" + String(Y_ENDSTOP_MIN);
    response += ",ymax=" + String(Y_ENDSTOP_MAX);
    response += ",zmin=" + String(Z_ENDSTOP_MIN);
    response += ",zmax=" + String(Z_ENDSTOP_MAX);
    response += ",home=" + String(atHomePosition ? 1 : 0);
    response += ",started=" + String(machineStarted ? 1 : 0);
    response += ">";
    Serial.println(response);
  
  // Motor control commands (only when stopped)
  } else if (Tempval1.startsWith("SetMotor,")) {
    if (machineStarted) {
      REPORT_ERROR(EC_CMD_STARTED, "Stop first");
      return;
    }
    
    int comma1 = Tempval1.indexOf(',');
    int comma2 = Tempval1.indexOf(',', comma1 + 1);
    String motor = Tempval1.substring(comma1 + 1, comma2);
    int state = Tempval1.substring(comma2 + 1).toInt();
    
    if (motor == "Xenable") digitalWrite(Xenable, state);
    else if (motor == "Yenable") digitalWrite(Yenable, state);
    else if (motor == "Zenable") digitalWrite(Zenable, state);
    else if (motor == "E0enable") digitalWrite(E0enable, state);
    else if (motor == "E1enable") digitalWrite(E1enable, state);
    else if (motor == "Vacuum1") digitalWrite(Vacuum1, state);
    else if (motor == "Vacuum2") digitalWrite(Vacuum2, state);
    else if (motor == "Lights") digitalWrite(Lights, state);
    
    Serial.println("<OK,Motor=" + motor + ",State=" + String(state) + ">");
  
  // Parameter update commands (only allowed at home position AND when stopped)
  } else if (Tempval1.startsWith("SetParam,")) {
    if (machineStarted) {
      REPORT_ERROR(EC_CMD_STARTED, "Stop first");
      return;
    }
    if (!atHomePosition) {
      REPORT_ERROR(EC_NOT_AT_HOME, "Home first");
      return;
    }
    
    int comma1 = Tempval1.indexOf(',');
    int comma2 = Tempval1.indexOf(',', comma1 + 1);
    String param = Tempval1.substring(comma1 + 1, comma2);
    int value = Tempval1.substring(comma2 + 1).toInt();
    
    if (param == "speed") speed = value;
    else if (param == "zspeed") zspeed = value;
    else if (param == "zespeed") zespeed = value;
    else if (param == "xcal") Xcal = value;
    else if (param == "ycal") Ycal = value;
    else if (param == "zcal") Zcal = value;
    else if (param == "pickup_thresh") pickup_threshold = value;
    else if (param == "release_thresh") release_threshold = value;
    else if (param == "hcc") HCC = value;
    else if (param == "ycc") YCourseCorrection = value;
    else if (param == "xcc") XCourseCorrection = value;
    
    Serial.println("<OK,Param=" + param + ",Value=" + String(value) + ">");
  
  // Query current parameters
  } else if (Tempval1 == "QueryParams") {
    String response = "<Params";
    response += ",speed=" + String(speed);
    response += ",zspeed=" + String(zspeed);
    response += ",zespeed=" + String(zespeed);
    response += ",xcal=" + String(Xcal);
    response += ",ycal=" + String(Ycal);
    response += ",zcal=" + String(Zcal);
    response += ",pickup_thresh=" + String(pickup_threshold);
    response += ",release_thresh=" + String(release_threshold);
    response += ",hcc=" + String(HCC);
    response += ",ycc=" + String(YCourseCorrection);
    response += ",xcc=" + String(XCourseCorrection);
    response += ">";
    Serial.println(response);
  
  } else if (match = 1){match = 0;} else {
    REPORT_ERROR(EC_UNKNOWN_CMD, Tempval1.substring(0,10));
    ForLoop(1, 34);
  }
}

//Loop through trays and assign values accordingly
void ForLoop(byte first, byte last) {
  for (byte i = first; i <= last; i++) {
    if (AssignedTrayValue[i] == "") {AssignedTrayValue[i] = Tempval1;PrintLCD("Tray assigned ", Tempval1);delay(10);} //if tray assignment is empty assign it the received variable
    if (AssignedTrayValue[i] == Tempval1 && CountArray[i] <= 375) {Tray(i);Serial.println((String) "Went to tray" + i);break;} //if tray assignment equals received value go to that tray and break the loop
    else if (i == 34 && CountArray[34] < 375) {Tray(34);break;} //If loop reaches 34 go to tray 34 if less than 375 cards is in that tray
    else if (i == 34 && CountArray[34] >= 375) {
      REPORT_ERROR(EC_OVERFLOW_FULL, "Tray 34 full");
      StopMachine(); Tempval1 = ""; break;
    }
  }
}

//Function to handle the tray action based on the given tray number
void Tray(short var) {
  atHomePosition = false;  //Mark as not at home during tray operation
  Move1(0, initial_pickup_distance, zspeed);ReadRange(1); //Move the Z-axis to the initial pickup position and read the range
  pick((range[0] + range[1]) / 2 * Zcal, 0);upcount++;CountArray[var]++; //Pick the card, Increment the total upcount and tray count
  short x, y;
  //Determine the X and Y coordinates based on the tray number
  for (byte j = 0; j < 6; j++) {if (X[j / 2][j % 4] == var) {x = xOffsets[j];break;}}
  for (byte j = 0; j < 4; j++) {if (Y[j][0] == var) {y = yOffsets[j];break;}}
  int absX = abs(Xcal * x);int absY = abs(Ycal * y);
  int moveToDirectionX =   (x >= 0) ? 1 : 0; int moveToDirectionY =   (y >= 0) ? 0 : 1; Move4(moveToDirectionX,   moveToDirectionY,   absX, absY); // Move to coordinate
  pick(initial_drop_distance, 1); ReadRange(5); while ((range[4] + range[5]) / 2 > 53) {Move1(1,5,zespeed);}
  int moveBackDirectionX = (x >= 0) ? 0 : 1; int moveBackDirectionY = (y >= 0) ? 1 : 0; Move4(moveBackDirectionX, moveBackDirectionY, absX, absY); // Move back to home tray
  Move1(1, initial_pickup_distance, zespeed); //Move Z back up to home position
  if (y != 0) { Move4(0, 0, 0, YCourseCorrection); } if (x != 0) { Move4(0, 1, XCourseCorrection, 0); } //Make course corrections if not 0
  if (upcount % HCC == 0 && upcount >= HCC / 2) { Homemachine(); } //If HCC count reached, rehome the machine
  atHomePosition = true;  //Back at home position
}

//Function to move in 4 directions based on the provided parameters
void Move4(boolean dir1, boolean dir3, short steps1, short steps2) {
  boolean dir2;
  if (dir1 == 0) {dir2 = 1;}else{dir2 = 0;}
  digitalWrite(Xdir, dir1); digitalWrite(E0dir, dir2); digitalWrite(Ydir, dir3); digitalWrite(E1dir, dir3);
  for (short i = 0; (i < steps1 || i < steps2); i++) { //Loop to move for the required number of steps in both directions
    if (i < steps1) {digitalWrite(Xstep, HIGH); digitalWrite(E0step, HIGH);}
    if (i < steps2) {digitalWrite(Ystep, HIGH); digitalWrite(E1step, HIGH);}
    delayMicroseconds(speed);
    digitalWrite(Xstep, LOW); digitalWrite(E0step, LOW); digitalWrite(Ystep, LOW); digitalWrite(E1step, LOW); //Deactivate step pins after each pulse
    delayMicroseconds(speed);
  }
}

//Function to move along the Z-axis with a specified direction and steps
void Move1(boolean dir, short steps, short speed1) {
  digitalWrite(Zdir, dir);
  for (short i = 0; i < steps; i++) { //Loop to perform the movement
    digitalWrite(Zstep, HIGH); delayMicroseconds(speed1);
    digitalWrite(Zstep,  LOW); delayMicroseconds(speed1);
  }
}

//Function to print a message to the LCD screen
void PrintLCD(String var1, String var2) {
  lcd.clear(); lcd.setCursor(0, 0); lcd.print(var1); lcd.setCursor(0, 1); lcd.print(var2);
}

//Function to read range values from the sensor
void ReadRange(byte var1) {
    range[var1 - 1] = vl.readRange(); delay(10); //Read range from sensor
    range[var1] = vl.readRange(); delay(10);//Read range from sensor
}