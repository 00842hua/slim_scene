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
    n=split($1, a, "_"); 
    ground_truth=$2;
    Category[ground_truth]; 
    predict=$3
    if(a[2]=="1")
    {
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
    else
    {
        if($2==$3 || labelmap[$3]==$2)
        {
            FP[ground_truth]+=1;
        }
        else
        {
            TN[ground_truth]+=1;
        }
    }
} 
END{
    totalr=0;
    totalp=0;
    for (k in Category)
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
        print k""gap"     \t"recall"       \t"precision"\t"length(k)
    } 
    print "Sum:     \t"totalr"       \t"totalp; 
    print "AVG:     \t"totalr/length(Category)"       \t"totalp/length(Category); 
    totalr=0;
    totalp=0;
    for (k in Category)
    {
        if(k ~ "^植物_植物$|饮品_红酒酒标|^书$|^机动车工程车_汽车$|饮品_白酒|^饮品_饮料$|饮品_纯净水")
        {
            gap=""; 
            if(length(k)<=3)
            {
                gap="  ";
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
            print "I"k""gap"     \t"recall"       \t"precision
        }
    }
    print "ISum:     \t"totalr"       \t"totalp;
    print "IAVG:     \t"totalr/7"       \t"totalp/7;
}
