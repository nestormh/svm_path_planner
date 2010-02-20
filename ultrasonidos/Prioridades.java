public class Prioridades {

String tramoPrioritario;
String tramoSecundario;


public Prioridades()
{
}

public Prioridades(String prioritario, String secundario)
{tramoPrioritario = prioritario;
 tramoSecundario = secundario;
}

public String dimePrioritario()
{return tramoPrioritario;
}

public String dimeSecundario()
{return tramoSecundario;
}

public void fijaPrioritario(String prioritario)
{tramoPrioritario = prioritario;
}
 
public void fijaSecundario(String secundario)
{ tramoSecundario = secundario;
}
   
 
}