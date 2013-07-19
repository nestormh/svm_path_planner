package sibtra.util;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.net.URL;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import sibtra.ui.defs.Motor;

/** Para obtener clases de en un paquete
 * obtenido de 
 * http://forums.sun.com/thread.jspa?threadID=341935                                 
 * http://www.javaworld.com/javaworld/javatips/jw-javatip113.html                    
 * http://snippets.dzone.com/posts/show/4831                                         
 * @author alberto
 *
 */
public abstract class ClasesEnPaquete {
	
	static List<File> dondeEncontrarPaquete(String packageName,ClassLoader classLoader) throws IOException {
		if(packageName==null)
			throw new IllegalArgumentException("Nombre de paquete no puede ser null\n");
		if(classLoader==null)
			throw new IllegalArgumentException("ClassLoader no puede ser null\n");
        String path = packageName.replace('.', '/');
        Enumeration<URL> resources = classLoader.getResources(path);
        List<File> dirs = new ArrayList<File>();
        while (resources.hasMoreElements()) {
            URL resource = resources.nextElement();
            File fa=new File(resource.getFile());
            if(fa.exists())
            	dirs.add(fa);
        }
        return dirs;
	}
	
	static List<File> dondeEncontrarPaquete(String packageName) throws IOException {
		return dondeEncontrarPaquete(packageName, Thread.currentThread().getContextClassLoader());
	}
	
	
    /**
     * Scans all classes accessible from the context class loader which belong to the given package and subpackages.
     *
     * @param packageName The base package
     * @return The classes
     * @throws ClassNotFoundException
     * @throws IOException
     */
    private static Class[] getClasses(String packageName, ClassLoader cl)
            throws ClassNotFoundException, IOException {
        List<File> dirs = dondeEncontrarPaquete(packageName,cl);
        ArrayList<Class> classes = new ArrayList<Class>();
        for (File directory : dirs) {
            classes.addAll(findClasses(directory, packageName, cl));
        }
        return classes.toArray(new Class[classes.size()]);
    }

    /**
     * Recursive method used to find all classes in a given directory and subdirs.
     *
     * @param directory   The base directory
     * @param packageName The package name for classes found inside the base directory
     * @return The classes
     * @throws ClassNotFoundException
     */
    private static List<Class> findClasses(File directory, String packageName, ClassLoader cl) throws ClassNotFoundException {
        List<Class> classes = new ArrayList<Class>();
        if (!directory.exists()) {
            return classes;
        }
        File[] files = directory.listFiles();
        for (File file : files) {
            if (file.isDirectory()) {
                assert !file.getName().contains(".");
                classes.addAll(findClasses(file, packageName + "." + file.getName(),cl));
            } else if (file.getName().endsWith(".class")) {
            	classes.add(cl.loadClass((packageName + '.' + file.getName().substring(0, file.getName().length() - 6))));
            }
        }
        return classes;
    }
    
    static File findFicheroClase(File directory, String nombreClase) {
    	String nomFichClase = nombreClase.substring(nombreClase.lastIndexOf('.')+1)+".class";
    	File fichClase=new File(directory,nomFichClase);
    	if(fichClase.exists())
    		return fichClase;
    	else
    		return null;
    }
    
    /**
     * Clases que implementan un determinado interface en un determinado paquete (y sus subpaquetes)
     * @param nombreInterface interface a implementar
     * @param nombrePaquete paquete donde buscar
     * @param cl cargador de clases a utilizar
     * @return array con las clases que lo cumplen, vacío so no hay ninguna
     */
    public static Class[] clasesImplementan(String nombreInterface, String nombrePaquete, ClassLoader cl) {
    	if(nombreInterface==null || nombreInterface.length()==0)
    		throw new IllegalArgumentException("El nombre del interface debe ser cadena no vacía");
    	if(nombrePaquete==null || nombrePaquete.length()==0)
    		throw new IllegalArgumentException("El nombre del paquete debe ser cadena no vacía");
        ArrayList<Class> clasCumple = new ArrayList<Class>();
        try {
			Class[] arrClas= ClasesEnPaquete.getClasses(nombrePaquete,cl);
			for(int i=0; i<arrClas.length; i++) {
				Class ca=arrClas[i];
				if(!ca.isInterface() && !ca.isMemberClass() && !ca.isLocalClass() && !ca.isAnonymousClass() 
						&& !ca.isPrimitive() && !Modifier.isAbstract(ca.getModifiers())
						) {
					//Tenemos que ver que implementa modulo
					Class[] intImp=ca.getInterfaces();
					boolean implementa=false;
					for(int j=0; !implementa && j<intImp.length; j++) {
						implementa=intImp[j].getName().equals(nombreInterface);
					}
					if(implementa) {
						clasCumple.add(ca);
					}
				}
			}
        } catch (ClassNotFoundException e) {
        	System.err.println("ClassNotFoundException:"+e.getMessage());
        	e.printStackTrace();
        } catch (IOException e) {
        	System.err.println("IOException:"+e.getMessage());
        	e.printStackTrace();
        }
        return clasCumple.toArray(new Class[clasCumple.size()]);
    }
    
    /**
     * Clases que implementan un determinado interface en un determinado paquete (y sus subpaquetes)
     * Se usa cargador de clases por defecto
     * @param nombreInterface interface a implementar
     * @param nombrePaquete paquete donde buscar
     * @return array con las clases que lo cumplen, vacío so no hay ninguna
     */
    public static Class[] clasesImplementan(String nombreInterface, String nombrePaquete) {
    	return clasesImplementan(nombreInterface, nombrePaquete,
    			Thread.currentThread().getContextClassLoader());
    }    
    public static void main(String[] args) {
    
    	try {
			Class[] arrClas= ClasesEnPaquete.getClasses("sibtra.ui.modulos"
					,Thread.currentThread().getContextClassLoader());
			
			System.out.println("Encontradas: ");
			for(int i=0; i<arrClas.length; i++) {
				Class ca=arrClas[i];
				System.out.println(ca.getName()+"\n\t"
						+"\tInterface:"+ca.isInterface()
						+"\tLocal:"+ca.isLocalClass()
						+"\tMember:"+ca.isMemberClass()
						+"\tAnonima:"+ca.isAnonymousClass()
						+"\tPrimitiva:"+ca.isPrimitive()
						+"\tAbstracta:"+Modifier.isAbstract(ca.getModifiers())
				);
				
				Class[] susCla=ca.getClasses();
				System.out.println("\tResultado getClasses:");
				for(int j=0; j<susCla.length;j++) {
					System.out.println("\t\t"+susCla[j].getName());
				}
				Class[] susInt=ca.getInterfaces();
				System.out.println("\tResultado getInterfaces:");
				for(int j=0; j<susInt.length;j++) {
					System.out.println("\t\t"+susInt[j].getName());
				}
			}

			System.out.println("Nos Interesan: ");
			for(int i=0; i<arrClas.length; i++) {
				Class ca=arrClas[i];
				if(!ca.isInterface() && !ca.isMemberClass() && !ca.isLocalClass() && !ca.isAnonymousClass() 
						&& !ca.isPrimitive() && !Modifier.isAbstract(ca.getModifiers())
						) {
					//Tenemos que ver que implementa modulo
					Class[] intImp=ca.getInterfaces();
					boolean implementa=false;
					for(int j=0; !implementa && j<intImp.length; j++) {
						implementa=intImp[j].getName().startsWith("sibtra.ui.modulos.");
					}
					if(implementa) {
						System.out.println(ca.getName());
						for(int j=0; j<intImp.length;j++) {
							System.out.println("\t\t"+intImp[j].getName());
						}
					}
				}
			}
    	
    	} catch (ClassNotFoundException e) {
			System.err.println("ClassNotFoundException:"+e.getMessage());
			e.printStackTrace();
		} catch (IOException e) {
			System.err.println("IOException:"+e.getMessage());
			e.printStackTrace();
		}
		char c='s';
		
		do {
		System.out.println("\n\n\nDirectamente:");
		{ 
			String intAct="sibtra.ui.modulos.CalculoDireccion";
			System.out.println("Interface:"+intAct);
			Class[] clasesImp=ClasesEnPaquete.clasesImplementan(intAct, "sibtra.ui.modulos");
			for(int i=0;i<clasesImp.length;i++)
				System.out.println("\t"+clasesImp[i].getName());
		}
		{ 
			String intAct="sibtra.ui.modulos.CalculoVelocidad";
			System.out.println("Interface:"+intAct);
			Class[] clasesImp=ClasesEnPaquete.clasesImplementan(intAct, "sibtra.ui.modulos");
			for(int i=0;i<clasesImp.length;i++)
				System.out.println("\t"+clasesImp[i].getName());
		}
		{ 
			String intAct="sibtra.ui.modulos.Motor";
			System.out.println("Interface:"+intAct);
			Class[] clasesImp=ClasesEnPaquete.clasesImplementan(intAct, "sibtra.ui.modulos");
			for(int i=0;i<clasesImp.length;i++)
				System.out.println("\t"+clasesImp[i].getName());
		}
		{ 
			String intAct="sibtra.ui.modulos.DetectaObstaculos";
			System.out.println("Interface:"+intAct);
			Class[] clasesImp=ClasesEnPaquete.clasesImplementan(intAct, "sibtra.ui.modulos");
			for(int i=0;i<clasesImp.length;i++)
				System.out.println("\t"+clasesImp[i].getName());
		}
		
		ClassLoader cargador=new CargadorDeModulos(CargadorDeModulos.class.getClassLoader(),"sibtra.ui.modulos");

		Class<Motor> motNew;
		Motor obMotor=null;
		try {
			Object obj=(cargador.loadClass("sibtra.ui.modulos.MotorNuevo").newInstance());
			Object obj2=(cargador.loadClass("sibtra.ui.modulos.MotorNuevo").newInstance());
			System.out.println("Clase:"+obj.getClass());
			obMotor=(Motor)obj;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("Motor nuevo es: "+obMotor.getNombre());
		
		System.out.append("Entra para repetir (q para salir");
			
		try {
			c=(char) System.in.read();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		} while (c!=(int)'q');
    }

    /** Dado el array de clases ivoca el metodo getNombre() de cada clase y devueve la respuesta en misma
     * posición del array de String resultado
     * @return array con los resultados de invorcar getNombre(). NULL para las clases que den algún problema.
     */
	public static String[] nombreClases(Class[] arrClas) {
		String[] resp=new String[arrClas.length];
		for(int i=0; i<arrClas.length; i++ ) {
				Class ca=arrClas[i];
				resp[i]=null; //se quedará así si hay algún problema
				try {
					Method mn=ca.getMethod("getNombre", (Class[])null);
					//instanciamos objeto con constructor vacio
					Object ob=arrClas[i].newInstance();
					String nombre=(String)mn.invoke(ob, (Object[])null);
					resp[i]=nombre;
				} catch (SecurityException e) {
					System.err.println("nombreClases: problema de seguridad al acceder a la clase "+ca.getName()
							+": "+e.getMessage());
				} catch (IllegalArgumentException e) {
					System.err.println("nombreClases: problema de argumentos al acceder a getNombre() de la clase "+ca.getName()
							+": "+e.getMessage());
				} catch (NoSuchMethodException e) {
					System.err.println("nombreClases: la clase "+ca.getName()+" no tiene el método getNombre()"
							+": "+e.getMessage());
				} catch (InstantiationException e) {
					System.err.println("nombreClases: la clase "+ca.getName()+" no se puede instanciar"
							+": "+e.getMessage());
				} catch (IllegalAccessException e) {
					System.err.println("nombreClases: no podemos acceder al método getNombre() de la clase "+ca.getName()
							+": "+e.getMessage());
				} catch (InvocationTargetException e) {
					System.err.println("nombreClases: invocación a getNombre() de la clase "+ca.getName()+" produjo excepción"
							+": "+e.getMessage());
				}
		}
		return resp;
	}

}
