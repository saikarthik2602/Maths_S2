format long g

 % FOR TESTING DATA SET
    
  Mat_1 = readmatrix('aps_failure_test_set.csv');
    
  Test = Mat_1(1:16000,2:171);
    
   Mean_Test = fillmissing(Test,'movmean',16000); 

   Median_Test = fillmissing(Test,'movmedian',16000);
  
   Near_Test = fillmissing(Test,'nearest');

   
   % IN FFILL WE NEED TO MAKE LAST ROW OF NAN VALUES WITH ZEROS.
   
   duplicate_test_ffill = Test;
   
LAST_ROW_TEST = duplicate_test_ffill(size(duplicate_test_ffill,1),:);

ffill_last_row = fillmissing(LAST_ROW_TEST,'constant',0);

duplicate_test_ffill(size(duplicate_test_ffill,1),:)=[];

duplicate_test_ffill(size(duplicate_test_ffill,1)+1,:) = ffill_last_row;

Ffill_Test = fillmissing(duplicate_test_ffill,'next');


duplicate_test_bfill = Test;

% VALUES WHICH ARE nan MUST BE REPLACED WITH ZERO IN THE FIRST ROW

FIRST_ROW_TEST = duplicate_test_bfill(1,:);

bfill_first_row = fillmissing(FIRST_ROW_TEST,'constant',0);

duplicate_test_bfill(1,:) = bfill_first_row;

Bfill_Test = fillmissing(duplicate_test_bfill,'previous');



% {{{}}}}}}}{{{{}}}}}}

Mat = readmatrix('aps_failure_training_set.csv');

Traning = Mat(1:60000,2:171);

Mean_Traning =  fillmissing(Traning,'movmean',60000);

% size(Mean_Traning,1)
% size(Mean_Traning,2)

Normal_Mean_Traning = Mean_Traning;

Mean_1 = mean(Normal_Mean_Traning);

for a=1:170
    mean_dash=Mean_1(a);
    caluclating = Normal_Mean_Traning(:,a);
    maxi = max(caluclating);
    mini = min(caluclating);
    for b=1:60000
        diff = maxi-mini;
        if(diff==0)
            diff=1;
        end
        Normal_Mean_Traning(b,a) = (abs(Normal_Mean_Traning(b,a)-mean_dash))/diff;
    end
end

Median_Traning = fillmissing(Traning,'movmedian',60000);

Normal_Median_Traning = Median_Traning;

Mean_Median = mean(Normal_Median_Traning);

for c=1:170
    mean_dash=Mean_Median(c);
    caluclating = Normal_Median_Traning(:,c);
    maxi = max(caluclating);
    mini = min(caluclating);
    for d=1:60000
        diff=maxi-mini;
        if(diff==0)
            diff=1;
        end
        Normal_Median_Traning(d,c) = (abs(Normal_Median_Traning(d,c)-mean_dash))/diff;
    end
end

Near_Traning = fillmissing(Traning,'nearest');

Normal_Near_Traning = Near_Traning;

Mean_Near = mean(Normal_Near_Traning);

for e=1:170
    mean_dash=Mean_Near(e);
    caluclating = Normal_Near_Traning(:,e);
    maxi = max(caluclating);
    mini = min(caluclating);
    for f=1:60000
        diff=maxi-mini;
        if(diff==0)
            diff=1;
        end
        Normal_Near_Traning(f,e) = (abs(Normal_Near_Traning(f,e)-mean_dash))/diff;
    end
        
end

duplicate_Ffill = Traning;

LAST_ROW = duplicate_Ffill(size(duplicate_Ffill,1),:);

filled_last_row = fillmissing(LAST_ROW,'constant',0);

duplicate_Ffill(size(duplicate_Ffill,1),:)=[];

duplicate_Ffill(size(duplicate_Ffill,1)+1,:) = filled_last_row;

Ffill_Traning = fillmissing(duplicate_Ffill,'next');

Normal_Ffill_Traning = Ffill_Traning;

Mean_Ffill = mean(Normal_Ffill_Traning);

for g=1:170
    mean_dash= Mean_Ffill(g);
    caluclating = Normal_Near_Traning(:,g);
    maxi = max(caluclating);
    mini = min(caluclating);
    for h=1:60000
        diff=maxi-mini;
        if(diff==0)
            diff=1;
        end
        Normal_Near_Traning(h,g) = abs(Normal_Near_Traning(h,g)-mean_dash)/diff;
    end
end


duplicate_Bfill = Traning;

FIRST_ROW = duplicate_Bfill(1,:);

filled_first_row = fillmissing(FIRST_ROW,'constant',0);

duplicate_Bfill(1,:) = filled_first_row;

Bfill_Traning = fillmissing(duplicate_Bfill,'previous');

Normal_Bfill_Traning = Bfill_Traning;

Mean_Bfill = mean(Bfill_Traning);

for i=1:170
    mean_dash = Mean_Bfill(i);
    caluclating = Bfill_Traning(:,i);
    maxi = max(caluclating);
    mini = min(caluclating);
    for j=1:60000
        diff = maxi-mini;
        if(diff==0)
            diff=1;
        end
        Normal_Bfill_Traning(j,i) = abs(Normal_Bfill_Traning(j,i)-mean_dash)/diff;
    end
end

Traning_s1 = [];
Traning_s2 = {};


Test_s1 = [];
Test_s2 = {};



for k=1:170
    
    disp('IN I TH ITERATION');
    disp(k);
    
    temp_values = []; 
    
    MEAN_1 = Normal_Mean_Traning;
    Y_1 = MEAN_1(:,k);
    MEAN_1(:,k)=[];
    ones_1 = ones(size(MEAN_1,1),1);
    MEAN_1(:,size(MEAN_1,2)+1) = ones_1;
    theta1 = (pinv(MEAN_1'*MEAN_1))*MEAN_1'*Y_1;
    
    duplicate_mean = Mean_Traning;
    duplicate_mean(:,k) = [];
    ones_1 = ones(size(duplicate_mean,1),1);
    duplicate_mean(:,size(duplicate_mean,2)+1) = ones_1;
    
    original_y = duplicate_mean*theta1;
    original_y = abs(original_y);
    y_c = Traning(:,k);
    
    RMSE_MEAN = 0;
    cnt1 = 0;
    for l=1:size(original_y,1)
        if(isnan(y_c(l,1)))
            continue
        else
            cnt1=cnt1+1;
            RMSE_MEAN = RMSE_MEAN+(original_y(l,1)-y_c(l,1))*(original_y(l,1)-y_c(l,1));
        end
    end
    
    RMSE_TRANING_MEAN = sqrt((RMSE_MEAN)/cnt1)
    
    temp_values(1) = RMSE_TRANING_MEAN ;
    
    MEDIAN_1 = Normal_Median_Traning;
    Y_2 = MEDIAN_1(:,k);
    MEDIAN_1(:,k) = [];
    ones_2 = ones(size(MEDIAN_1,1),1);
    MEDIAN_1(:,size(MEDIAN_1,2)+1) = ones_2;
    theta2 = (pinv(MEDIAN_1'*MEDIAN_1))*MEDIAN_1'*Y_2;
    
    duplicate_median = Median_Traning;
    duplicate_median(:,k) = [];
    ones_2 = ones(size(duplicate_median,1),1);
    duplicate_median(:,size(duplicate_median,2)+1) = ones_2;
    
    original_y1 = duplicate_median*theta2;
    original_y1 =  abs(original_y1);
    y_c1 = Traning(:,k);
    
    RMSE_MEDIAN = 0;
    cnt2 = 0;
    
    for m=1:size(original_y1,1)
        if(isnan(y_c1(m,1)))
            continue
        else
            cnt2=cnt2+1;
            RMSE_MEDIAN = RMSE_MEDIAN+(original_y1(m,1)-y_c1(m,1))*(original_y1(m,1)-y_c1(m,1));
        end
    end
    RMSE_TRANING_MEDIAN = sqrt((RMSE_MEDIAN)/cnt2)
    
    temp_values(2) = RMSE_TRANING_MEDIAN;
    
    NEAR_1 = Normal_Near_Traning;
    Y_3 = NEAR_1(:,k);
    NEAR_1(:,k) = [];
    ones_3 = ones(size(NEAR_1,1),1);
    NEAR_1(:,size(NEAR_1,2)+1) = ones_3;
    theta3 = (pinv(NEAR_1'*NEAR_1))*NEAR_1'*Y_3;
    
    duplicate_near = Near_Traning;
    duplicate_near(:,k) = [];
    ones_3 = ones(size(duplicate_near,1),1);
    duplicate_near(:,size(duplicate_near,2)+1) = ones_3;
    
    original_y2 = duplicate_near*theta3;
    original_y2 =  abs(original_y2);
    y_c2 = Traning(:,k);
    
    RMSE_NEAR = 0;
    cnt3 = 0;
    
    for n=1:size(original_y2,1)
        if(isnan(y_c2(n,1)))
            continue
        else
            cnt3=cnt3+1;
            RMSE_NEAR = RMSE_NEAR+(original_y2(n,1)-y_c2(n,1))*(original_y2(n,1)-y_c2(n,1));
        end
    end
    RMSE_TRANING_NEAR = sqrt((RMSE_NEAR)/cnt3)
    
    temp_values(3) = RMSE_TRANING_NEAR;
    
    BFILL_1 = Normal_Bfill_Traning;
    Y_4 = BFILL_1(:,k);
    BFILL_1(:,k) = [];
    ones_4 = ones(size(BFILL_1,1),1);
    BFILL_1(:,size(BFILL_1,2)+1) = ones_4;
    theta4 = (pinv(BFILL_1'*BFILL_1))*BFILL_1'*Y_4;
    
    duplicate_bfill = Bfill_Traning;
    duplicate_bfill(:,k) = [];
    ones_4 = ones(size(duplicate_bfill,1),1);
    duplicate_bfill(:,size(duplicate_bfill,2)+1) = ones_4;
    
    original_y3 = duplicate_bfill*theta4;
    original_y3 =  abs(original_y3);
    y_c3 = Traning(:,k);
    
    RMSE_BFILL = 0;
    cnt4 = 0;
    
    for p=1:size(original_y3,1)
        if(isnan(y_c3(p,1)))
            continue
        else
            cnt4=cnt4+1;
            RMSE_BFILL = RMSE_BFILL+(original_y3(p,1)-y_c3(p,1))*(original_y3(p,1)-y_c3(p,1));
        end
    end
    RMSE_TRANING_BFILL = sqrt((RMSE_BFILL)/cnt4)
    
    temp_values(4) = RMSE_TRANING_BFILL;
    
    FFILL_1 = Normal_Ffill_Traning;
    Y_5 = FFILL_1(:,k);
    FFILL_1(:,k) = [];
    ones_5= ones(size(FFILL_1,1),1);
    FFILL_1(:,size(FFILL_1,2)+1) = ones_5;
    theta5 = (pinv(FFILL_1'*FFILL_1))*FFILL_1'*Y_5;
    
    duplicate_ffill = Ffill_Traning;
    duplicate_ffill(:,k) = [];
    ones_5 = ones(size(duplicate_ffill,1),1);
    duplicate_ffill(:,size(duplicate_ffill,2)+1) = ones_5;
    
    original_y4 = duplicate_ffill*theta5;
    original_y4 =  abs(original_y4);
    y_c4 = Traning(:,k);
    
    RMSE_FFILL = 0;
    cnt5 = 0;
    
    
    for q=1:size(original_y4,1)
        if(isnan(y_c4(q,1)))
            continue
        else
            cnt5=cnt5+1;
            RMSE_FFILL = RMSE_FFILL+(original_y4(q,1)-y_c4(q,1))*(original_y4(q,1)-y_c4(q,1));
        end
    end
    RMSE_TRANING_FFILL = sqrt((RMSE_FFILL)/cnt5)
    
     
    temp_values(5) = RMSE_TRANING_FFILL;

    
    mini = min(temp_values);
    
    Traning_s1(k) = mini;
    
    if(mini==temp_values(1))
        Traning_s2{k} = 'MEAN';
    end
    if(mini==temp_values(2))
            Traning_s2{k} = 'MEDIAN';
    end
    if(mini==temp_values(3))
            Traning_s2{k} = 'NEAR';
    end
    if(mini==temp_values(4))
            Traning_s2{k} = 'BFILL';
    end
    if(mini==temp_values(5))
            Traning_s2{k} = 'FFIll';
    end
    
    % TESTING DATA^^^
    
    test_values = [];
    
    duplicate_mean_test = Mean_Test;
    duplicate_mean_test(:,k) = [];
    test_ones_1 = ones(size(duplicate_mean_test,1),1);
    duplicate_mean_test(:,size(duplicate_mean_test,2)+1) = test_ones_1;
    
    test_y1 = duplicate_mean_test*theta1;
    test_y1 = abs(test_y1);
    original_1 = Test(:,k);
    RMSE_1 = 0;
    for r=1:size(test_y1,1)
        if(isnan(original_1(r,1)))
            continue;
        else
            RMSE_1 = RMSE_1+(test_y1(r,1)-original_1(r,1))*(test_y1(r,1)-original_1(r,1));
        end
        
    end
    
    RMSE_TEST_MEAN = sqrt((RMSE_1)/cnt1)
    
    test_values(1) = RMSE_TEST_MEAN;
    
    duplicate_median_test = Median_Test;
    duplicate_median_test(:,k) = [];
    test_ones_2 = ones(size(duplicate_median_test,1),1);
    duplicate_median_test(:,size(duplicate_median_test,2)+1) = test_ones_2;
    
    test_y2 = duplicate_median_test*theta2;
    test_y2 = abs(test_y2);
    original_2 = Test(:,k);
    RMSE_2 = 0;
    for s=1:size(test_y2,1)
        if(isnan(original_2(s,1)))
            continue;
        else
            RMSE_2 = RMSE_2+(test_y2(s,1)-original_2(s,1))*(test_y2(s,1)-original_2(s,1));
        end
    end
    
    RMSE_TEST_MEDIAN = sqrt((RMSE_2)/cnt2)
    
    test_values(2) = RMSE_TEST_MEDIAN;
    
    duplicate_near_test = Near_Test;
    duplicate_near_test(:,k) = [];
    test_ones_3 = ones(size(duplicate_near_test,1),1);
    duplicate_near_test(:,size(duplicate_near_test,2)+1) = test_ones_3;
    
    
    test_y3 = duplicate_median_test*theta3;
    test_y3 = abs(test_y3);
    original_3 = Test(:,k);
    RMSE_3 = 0;
    for t=1:size(test_y3,1)
        if(isnan(original_3(t,1)))
            continue;
        else
            RMSE_3 = RMSE_3+(test_y3(t,1)-original_3(t,1))*(test_y3(t,1)-original_3(t,1));
        end
    end
    
    RMSE_TEST_NEAR = sqrt((RMSE_3)/cnt3)
    
    test_values(3) = RMSE_TEST_NEAR;
    
    duplicate_bfill_test = Bfill_Test;
    duplicate_bfill_test(:,k) = [];
    test_ones_4 = ones(size(duplicate_bfill_test,1),1);
    duplicate_bfill_test(:,size(duplicate_bfill_test,2)+1) = test_ones_4;
    
    
    test_y4 = duplicate_median_test*theta4;
    test_y4 = abs(test_y4);
    original_4 = Test(:,k);
    RMSE_4 = 0;
    for u=1:size(test_y4,1)
        if(isnan(original_4(u,1)))
            continue;
        else
            RMSE_4 = RMSE_4+(test_y4(u,1)-original_4(u,1))*(test_y4(u,1)-original_4(u,1));
        end
    end
    
    RMSE_TEST_BFILL = sqrt((RMSE_4)/cnt4)
    
    test_values(4) = RMSE_TEST_BFILL;
    
    duplicate_ffill_test = Ffill_Test;
    duplicate_ffill_test(:,k) = [];
    test_ones_5 = ones(size(duplicate_ffill_test,1),1);
    duplicate_ffill_test(:,size(duplicate_ffill_test,2)+1) = test_ones_5;
    
    
    test_y5 = duplicate_median_test*theta5;
    test_y5 = abs(test_y5);
    original_5 = Test(:,k);
    RMSE_5 = 0;
    for v=1:size(test_y5,1)
        if(isnan(original_5(v,1)))
            continue;
        else
            RMSE_5 = RMSE_5+(test_y5(v,1)-original_5(v,1))*(test_y5(v,1)-original_5(v,1));
        end
    end
    
    RMSE_TEST_FFILL = sqrt((RMSE_5)/cnt5)
    
    
    test_values(5) = RMSE_TEST_FFILL;
     
    mini1 = min(test_values);
    
    Test_s1(k) = mini1;
    
    if(mini1==test_values(1))
        Test_s2{k} = 'MEAN';
    end
    if(mini1==test_values(2))
            Test_s2{k} = 'MEDIAN';
    end
    if(mini1==test_values(3))
            Test_s2{k} = 'NEAR';
    end
    if(mini1==test_values(4))
            Test_s2{k} = 'BFILL';
    end
    if(mini1==test_values(5))
            Test_s2{k} = 'FFIll';
    end
    
%     if(k==10)
%         break;
%     end
    
     disp('NEXT ITERATION')
end


     disp('THE INFORMATION OF TRAINING SET')
     disp(Traning_s1);
     disp(Traning_s2);
     
     disp('THE INFORMATION OF TESTING SET')
     disp(Test_s1);
     disp(Test_s2);