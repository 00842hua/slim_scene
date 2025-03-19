BEGIN{
    counter=0;
    cols=10;
    print "<html><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" /><style> img {border:0px; margin:5px 5px; padding:0px 0px; max-width:150px; max-height:150px;} .divcss5{text-align:center} </style></head><table>";
}
{
    counter+=1;
    if((counter-1)%cols==0){print "<tr style=\"vertical-align:top\">"}
    predict_result = sprintf("<br/>%.2f", $2);
    line=sprintf("<td width=\"150\"><img src=\"%s\"/><br/><label>%s</label><td>", $1, predict_result);
    print line;
    if(counter%cols==0){print "</tr>"}
}
END{
    if(counter%cols!=0)
    {
        print "</tr>"
    }
    print "</table></html> ";
}