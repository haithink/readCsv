BEGIN{
FS=","
name=""
cnt = 0
}
{
	if($1 != name)
	{
	 name = $1
	 cnt++
	}

	if(cnt >= 201)
	{
		print $0
	}
}