format long g

% Testing data set


% ----- TO EXPLORE THE DATA IN THE DATA SET -----

who
    
whos

opts = detectImportOptions('aps_failure_test_set.csv');

preview('aps_failure_test_set.csv',opts);

% --------

 Mat_1 = readmatrix('aps_failure_test_set.csv');
    
  Test = Mat_1(1:16000,2:171);
    
   Mean_Test = fillmissing(Test,'movmean',16000); 

   Median_Test = fillmissing(Test,'movmedian',16000);
  
   Near_Test = fillmissing(Test,'nearest');

   duplicate_test_ffill = Test;
   
LAST_ROW_TEST = duplicate_test_ffill(size(duplicate_test_ffill,1),:);

ffill_last_row = fillmissing(LAST_ROW_TEST,'constant',0);

duplicate_test_ffill(size(duplicate_test_ffill,1),:)=[];

duplicate_test_ffill(size(duplicate_test_ffill,1)+1,:) = ffill_last_row;

Ffill_Test = fillmissing(duplicate_test_ffill,'next');


duplicate_test_bfill = Test;

FIRST_ROW_TEST = duplicate_test_bfill(1,:);

bfill_first_row = fillmissing(FIRST_ROW_TEST,'constant',0);

duplicate_test_bfill(1,:) = bfill_first_row;

Bfill_Test = fillmissing(duplicate_test_bfill,'previous');


% ---- TO CHECK IF THE VALUES IN THE DATASET AFTER FILLING WITH STATISTICAL
% METHODS HAS NO NAN ----

AAAA=any(isnan(Mean_Test),1);

BBBB=any(isnan(Median_Test),1);

CCCC = any(isnan(Near_Test),1);

DDDD = any(isnan(Ffill_Test),1);

EEEE = any(isnan(Bfill_Test),1);


% ------ ANOTHER WAY CHECK NAN VALUES IN THE DATA SET

% [rows, columns] = find(isnan(Mean_Test));

% N = unique(columns);

% ------


%Traning data set  ____||||_____$$$$___^^^^____  



Mat = readmatrix('aps_failure_training_set.csv');

% ----- TO EXPLORE THE DATA IN THE DATA SET -----

who
    
whos

opts = detectImportOptions('aps_failure_training_set.csv');

preview('aps_failure_training_set.csv',opts);

% --------

Traning = Mat(1:60000,2:171);

Mean_Traning =  fillmissing(Traning,'movmean',60000);

Median_Traning = fillmissing(Traning,'movmedian',60000);

Near_Traning = fillmissing(Traning,'nearest');

duplicate_Ffill = Traning;

LAST_ROW = duplicate_Ffill(size(duplicate_Ffill,1),:);

filled_last_row = fillmissing(LAST_ROW,'constant',0);

duplicate_Ffill(size(duplicate_Ffill,1),:)=[];

duplicate_Ffill(size(duplicate_Ffill,1)+1,:) = filled_last_row;

Ffill_Traning = fillmissing(duplicate_Ffill,'next');

duplicate_Bfill = Traning;

FIRST_ROW = duplicate_Bfill(1,:);

filled_first_row = fillmissing(FIRST_ROW,'constant',0);

duplicate_Bfill(1,:) = filled_first_row;

Bfill_Traning = fillmissing(duplicate_Bfill,'previous');


% ---- TO CHECK IF THE VALUES IN THE DATASET AFTER FILLING WITH STATISTICAL
% METHODS HAS NO NAN ----

AAAA1 = any(isnan(Mean_Traning),1);

BBBB1 = any(isnan(Median_Traning),1);

CCCC1 = any(isnan(Near_Traning),1);

DDDD1 = any(isnan(Ffill_Traning),1);

EEEE1 = any(isnan(Bfill_Traning),1);


Traning_s1 = [];
Traning_s2 = {};


Test_s1 = [];
Test_s2 = {};


OUTPUT_TRAINING = zeros(60000,170);

OUTPUT_TEST = zeros(16000,170);


for i=1:170
    
    disp('In the i th iteration');
    disp(i);
    
    temp_values = [];
    
    duplicate_1 = Mean_Traning;
    Y_1 = duplicate_1(:,i);
    duplicate_1(:,i) = [];
    ones_1 = ones(size(duplicate_1,1),1);
    duplicate_1(:,size(duplicate_1,2)+1) = ones_1;
    theta1 = (pinv(duplicate_1'*duplicate_1))*duplicate_1'*Y_1;
    
    yc1 = duplicate_1*theta1;
    yc1=abs(yc1);
    original_y1 = Traning(:,i);
    
    cnt1=0;
    RMSE_1 = 0;
    for j=1:size(yc1,1)
        if(isnan(original_y1(j,1)))
            continue;
        else
            cnt1=cnt1+1;
            RMSE_1 = RMSE_1+(yc1(j,1)-original_y1(j,1))*(yc1(j,1)-original_y1(j,1));
        end
    end
    RMSE_MEAN_TRAINING = sqrt(RMSE_1/cnt1);
    
    temp_values(1) = RMSE_MEAN_TRAINING;
    
    duplicate_2 = Median_Traning;
    Y_2 = duplicate_2(:,i);
    duplicate_2(:,i) = [];
    ones_2 = ones(size(duplicate_2,1),1);
    duplicate_2(:,size(duplicate_2,2)+1) = ones_2;
    theta2 = (pinv(duplicate_2'*duplicate_2))*duplicate_2'*Y_2;
    
    yc2 = duplicate_2*theta2;
    yc2=abs(yc2);
    original_y2 = Traning(:,i);
    
    cnt2=0;
    RMSE_2 = 0;
    for k=1:size(yc2,1)
        if(isnan(original_y2(k,1)))
            continue;
        else
            cnt2=cnt2+1;
            RMSE_2 = RMSE_2+(yc2(k,1)-original_y2(k,1))*(yc2(k,1)-original_y2(k,1));
        end
    end
    
    RMSE_MEDIAN_TRAINING = sqrt(RMSE_2/cnt2);
    
    temp_values(2) = RMSE_MEDIAN_TRAINING;
    
    duplicate_3 = Near_Traning;
     Y_3 = duplicate_3(:,i);
    duplicate_3(:,i) = [];
    ones_3 = ones(size(duplicate_3,1),1);
    duplicate_3(:,size(duplicate_3,2)+1) = ones_3;
    theta3 = (pinv(duplicate_3'*duplicate_3))*duplicate_3'*Y_3;
    
    yc3 = duplicate_3*theta3;
    yc3=abs(yc3);
    original_y3 = Traning(:,i);
    
    cnt3=0;
    RMSE_3 = 0;
    for l=1:size(yc3,1)
        if(isnan(original_y3(l,1)))
            continue;
        else
            cnt3=cnt3+1;
            RMSE_3 = RMSE_3+(yc3(l,1)-original_y3(l,1))*(yc3(l,1)-original_y3(l,1));
        end
    end
    
    RMSE_NEAR_TRAINING = sqrt(RMSE_3/cnt3);
    
    temp_values(3) = RMSE_NEAR_TRAINING;
    
    
    duplicate_4 = Bfill_Traning;
     Y_4 = duplicate_4(:,i);
    duplicate_4(:,i) = [];
    ones_4 = ones(size(duplicate_4,1),1);
    duplicate_4(:,size(duplicate_4,2)+1) = ones_4;
    theta4 = (pinv(duplicate_4'*duplicate_4))*duplicate_4'*Y_4;
    
    
    yc4 = duplicate_4*theta4;
    yc4=abs(yc4);
    original_y4 = Traning(:,i);
    
    cnt4=0;
    RMSE_4 = 0;
    for m=1:size(yc4,1)
        if(isnan(original_y4(m,1)))
            continue;
        else
            cnt4=cnt4+1;
            RMSE_4 = RMSE_4+(yc4(m,1)-original_y4(m,1))*(yc4(m,1)-original_y4(m,1));
        end
    end
    
    RMSE_BFILL_TRAINING = sqrt(RMSE_4/cnt4);
    
    temp_values(4) = RMSE_BFILL_TRAINING;
    
    
    duplicate_5 = Ffill_Traning;
    Y_5 = duplicate_5(:,i);
    duplicate_5(:,i) = [];
    ones_5 = ones(size(duplicate_5,1),1);
    duplicate_5(:,size(duplicate_5,2)+1) = ones_5;
    theta5 = (pinv(duplicate_5'*duplicate_5))*duplicate_5'*Y_5;
    
    
    yc5 = duplicate_5*theta5;
    yc5=abs(yc5);
    original_y5 = Traning(:,i);
    
    cnt5=0;
    RMSE_5 = 0;
    for n=1:size(yc5,1)
        if(isnan(original_y5(m,1)))
            continue;
        else
            cnt5=cnt5+1;
            RMSE_5 = RMSE_5+(yc5(m,1)-original_y5(m,1))*(yc5(m,1)-original_y5(m,1));
        end
    end
    
    RMSE_FFILL_TRAINING = sqrt(RMSE_5/cnt5);
    
    temp_values(5) = RMSE_FFILL_TRAINING;
    mini = min(temp_values);
    
    disp(mini)
    
    Traning_s1(i) = mini;
    
    if(mini==temp_values(1))
        Traning_s2{i} = 'MEAN';
    elseif(mini==temp_values(2))
            Traning_s2{i} = 'MEDIAN';
    
    elseif(mini==temp_values(3))
            Traning_s2{i} = 'NEAR';
    
    elseif(mini==temp_values(4))
            Traning_s2{i} = 'BFILL';
            
    elseif(mini==temp_values(5))
            Traning_s2{i} = 'FFIll';
    end
    
    
    % SECOND SET OF DATA
    
    test_values = [];
    
    duplicate1 = Mean_Test;
    Y1 = duplicate1(:,i);
    duplicate1(:,i) = [];
    ones1 = ones(size(duplicate1,1),1);
    duplicate1(:,size(duplicate1,2)+1) = ones1;
    
    
    yc_1 = duplicate1*theta1;
    yc_1=abs(yc_1);
    originaly1 = Test(:,i);
    
    RMSE1 = 0;
    for p=1:size(yc_1,1)
        if(isnan(originaly1(p,1)))
            continue;
        else
            RMSE1 = RMSE1+(yc_1(p,1)-originaly1(p,1))*(yc_1(p,1)-originaly1(p,1));
        end
    end
    RMSE_MEAN_TEST = sqrt(RMSE1/cnt1);
    
    test_values(1) = RMSE_MEAN_TEST;
    
    duplicate2 = Median_Test;
    Y2 = duplicate2(:,i);
    duplicate2(:,i) = [];
    ones2 = ones(size(duplicate2,1),1);
    duplicate2(:,size(duplicate2,2)+1) = ones2;
    
    
    yc_2 = duplicate2*theta2;
    yc_2=abs(yc_2);
    originaly2 = Test(:,i);
    
    RMSE2 = 0;
    for q=1:size(yc_2,1)
        if(isnan(originaly2(q,1)))
            continue;
        else
            RMSE2 = RMSE2+(yc_2(q,1)-originaly2(q,1))*(yc_2(q,1)-originaly2(q,1));
        end
    end
    RMSE_MEDIAN_TEST = sqrt(RMSE2/cnt2);
    
    test_values(2) = RMSE_MEDIAN_TEST;
    
    
    
    duplicate3 = Near_Test;
    Y3 = duplicate3(:,i);
    duplicate3(:,i) = [];
    ones3 = ones(size(duplicate3,1),1);
    duplicate3(:,size(duplicate3,2)+1) = ones3;
    
    
    yc_3 = duplicate3*theta3;
    yc_3=abs(yc_3);
    originaly3 = Test(:,i);
    
    RMSE3 = 0;
    for r=1:size(yc_3,1)
        if(isnan(originaly3(r,1)))
            continue;
        else
            RMSE3 = RMSE3+(yc_3(r,1)-originaly3(r,1))*(yc_3(r,1)-originaly3(r,1));
        end
    end
    RMSE_NEAR_TEST = sqrt(RMSE3/cnt3);
    
    test_values(3) = RMSE_NEAR_TEST;
    
    
   
    
    
    duplicate4 = Bfill_Test;
    Y4 = duplicate4(:,i);
    duplicate4(:,i) = [];
    ones4 = ones(size(duplicate4,1),1);
    duplicate4(:,size(duplicate4,2)+1) = ones4;
    
    
    yc_4 = duplicate4*theta4;
    yc_4=abs(yc_4);
    originaly4 = Test(:,i);
    
    RMSE4 = 0;
    for s=1:size(yc_4,1)
        if(isnan(originaly4(s,1)))
            continue;
        else
            RMSE4 = RMSE4+(yc_4(s,1)-originaly4(s,1))*(yc_4(s,1)-originaly4(s,1));
        end
    end
    RMSE_BFILL_TEST = sqrt(RMSE4/cnt4);
    
    test_values(4) = RMSE_BFILL_TEST;
    
    
    
    duplicate5 = Ffill_Test;
    Y5 = duplicate5(:,i);
    duplicate5(:,i) = [];
    ones5 = ones(size(duplicate5,1),1);
    duplicate5(:,size(duplicate5,2)+1) = ones5;
    
    
    yc_5 = duplicate5*theta5;
    yc_5=abs(yc_5);
    originaly5 = Test(:,i);
    
    RMSE5 = 0;
    for t=1:size(yc_5,1)
        if(isnan(originaly5(t,1)))
            continue;
        else
            RMSE5 = RMSE5+(yc_5(t,1)-originaly5(t,1))*(yc_5(t,1)-originaly5(t,1));
        end
    end
    
    RMSE_FFILL_TEST = sqrt(RMSE5/cnt5) ;
    
    test_values(5) = RMSE_FFILL_TEST;
    
    mini1 = min(test_values);
    
    disp(mini1)
    
    Test_s1(i) = mini1;
    
    if(mini1==test_values(1))
        Test_s2{i} = 'MEAN';
        OUTPUT_TEST(:,i) = yc_1;
        OUTPUT_TRAINING(:,i) = yc1;
        
    elseif(mini1==test_values(2))
            Test_s2{i} = 'MEDIAN';
            OUTPUT_TEST(:,i) = yc_2;
            OUTPUT_TRAINING(:,i) = yc2;
            
    elseif(mini1==test_values(3))
            Test_s2{i} = 'NEAR';
            OUTPUT_TEST(:,i) = yc_3;
            OUTPUT_TRAINING(:,i) = yc3;
            
    elseif(mini1==test_values(4))
            Test_s2{i} = 'BFILL';
            OUTPUT_TEST(:,i) = yc_4;
            OUTPUT_TRAINING(:,i) = yc4;

    elseif(mini1==test_values(5))
            Test_s2{i} = 'FFIll';
            OUTPUT_TEST(:,i) = yc_5;
            OUTPUT_TRAINING(:,i) = yc5;
    end
    
    disp('NEXT ITERATION');
    
    
%     if(i==10)
%         break;
%     end
end
     disp('THE INFORMATION OF TRAINING SET')
     disp(Traning_s1);
     disp(Traning_s2);
     
     disp('THE INFORMATION OF TESTING SET')
     disp(Test_s1);
     disp(Test_s2);
     
     
    
  Output_test = Mat_1(1:16000,2:171);
     
     for i=1:16000
         for j=1:170
             if(isnan(Output_test(i,j)))
                 WRITE = OUTPUT_TEST(i,j);
                 Output_test(i,j) = WRITE;
                 % disp(['filling in ',num2str(i),'th row and ',num2str(j),'th column Testing set'])
             end
         end
     end
     
  check_1 = any(isnan(Output_test),1)
  
  writematrix(Output_test,'Test_ouput.csv'); 
     
  Output_training = Mat(1:60000,2:171);
  
  
  for i=1:60000
         for j=1:170
             if(isnan(Output_training(i,j)))
                 WR = OUTPUT_TRAINING(i,j);
                 Output_training(i,j) = WR; 
                 % disp(['filling in ',num2str(i),'th row and ',num2str(j),'th column Training set'])
             end
         end
  end
     
   check_2 = any(isnan(Output_training),1)
     
   writematrix(Output_training,'Training_output.csv');
   
