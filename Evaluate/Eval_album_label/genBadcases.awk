NR==FNR{
    for(i=1; i<=NF; i++)
    {
        for (j=i+1; j<=NF; j++)
        {
            if (labelmap[$i]=="")
            {
                labelmap[$i]=$j
            }
            else
            {
                labelmap[$i] = labelmap[$i]"|"$j
            }
            if (labelmap[$j]=="")
            {
                labelmap[$j]=$i
            }
            else
            {
                labelmap[$j] = labelmap[$j]"|"$i
            }
        }
    }
}
NR>FNR{
    if ($2 == $3)
    {
    }
    else{
        n = split(labelmap[$3], a, "|")
        found = 0
        for (k in a)
        {
            if ($2 == a[k])
            {
                found=1;
                break;
            }
        }
        if (found == 0)
        {
            print $0
        }
    }
}
