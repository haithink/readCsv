BEGIN{
FS=","
}

{
 print $1 "," int($3+0.5) " " int($2+0.5) " " int($5+0.5) " " int($4+0.5)
}