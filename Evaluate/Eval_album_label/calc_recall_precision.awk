NR==FNR{
    for(i=1; i<=NF; i++)
    {
        for (j=i+1; j<=NF; j++)
        {
            labelmap[$i]=$j
            labelmap[$j]=$i
        }
    }
} 
NR>FNR{
    ground_truth=$2;
    predict=$3
    Category_GT[ground_truth];
    Category_predict[predict];
    if(ground_truth==predict || labelmap[predict]==ground_truth)
    {
        TP[ground_truth]+=1;
    }
    else
    {
        FN[ground_truth]+=1;
        FP[predict]+=1
    }

} 
END{
    totalr=0;
    totalp=0;
    for (k in Category_GT)
    {
        gap="      "; 
        if(length(k)<=6)
        {
            gap="              ";
        }
        if(length(k)<=3)
        {
            gap="                  ";
        }
        if (TP[k]+FN[k] == 0)
        {
            recall=0; 
        }
        else
        {
            recall=100*TP[k]/(TP[k]+FN[k]); 
        }
        if (TP[k]+FP[k] == 0)
        {
            precision=0; 
        }
        else
        {
            precision=100*TP[k]/(TP[k]+FP[k]); 
        }
        totalr+=recall; 
        totalp+=precision; 
        print k" "gap"     \t"recall"       \t"precision
    } 
    # print "Sum:     \t\t\t"totalr"       \t"totalp; 
    print "AVG:     \t\t\t"totalr/length(Category_GT)"       \t"totalp/length(Category_GT); 
    print "------------------------------------------"
    print "Recall Error"
    for (k in Category_predict)
    {
        if (k in Category_GT == 0)
        {
            gap="      "; 
            if(length(k)<=6)
            {
                gap="              ";
            }
            if(length(k)<=3)
            {
                gap="                  ";
            }
            print k" "gap"     \t"FP[k]
        }
    }
#    print "------------------------------------------"
#    
#    totalr=0;
#    totalp=0;
#    for (k in Category_GT)
#    {
#        if(k ~ "宠物_狗|宠物_猫")
#        {
#            gap="      "; 
#            if(length(k)<=6)
#            {
#                gap="              ";
#            }
#            if(length(k)<=3)
#            {
#                gap="                  ";
#            }
#            if (TP[k]+FN[k] == 0)
#            {
#                recall=0; 
#            }
#            else
#            {
#                recall=100*TP[k]/(TP[k]+FN[k]); 
#            }
#            if (TP[k]+FP[k] == 0)
#            {
#                precision=0; 
#            }
#            else
#            {
#                precision=100*TP[k]/(TP[k]+FP[k]); 
#            }
#            totalr+=recall; 
#            totalp+=precision; 
#            print "I"k" "gap"     \t"recall"       \t"precision
#        }
#    }
#    print "ISum:     \t\t\t"totalr"       \t"totalp;
#    print "IAVG:     \t\t\t"totalr/2"       \t"totalp/2;
}
