/**
 * 
 */
package sibtra.util;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.util.List;

/**
 * @author alberto
 *
 */
/**
 * Sacado de <a href="http://tutorials.jenkov.com/java-reflection/dynamic-class-loading-reloading.html">
 * aqui</a>
 * @author alberto
 *
 */
public class CargadorDeModulos extends ClassLoader {

	String packageName=null;
	List<File> dondeEncontrarla=null;

	public CargadorDeModulos(ClassLoader parent, String nombrePaquete) {
		super(parent);
		packageName=nombrePaquete;
		try {
		dondeEncontrarla=ClasesEnPaquete.dondeEncontrarPaquete(packageName);
		} catch (IOException e) {
			throw new IllegalArgumentException("El paquete "+packageName+" no se encuentra.");
		}
	}

	public Class loadClass(String nombreClase) throws ClassNotFoundException {
		//Si la clase pedida no está en el paquete soportado, la gestiona el padre
		if(!nombreClase.startsWith(packageName))
			return super.loadClass(nombreClase);
		Class clase=findLoadedClass(nombreClase);
		if(clase!=null)
			//Si ya esta cardada la devolvemos
			return clase;
		try {
		
			File fichClase=null;
			for(File da : dondeEncontrarla)
				if((fichClase=ClasesEnPaquete.findFicheroClase(da, nombreClase))!=null)
					break;
			if(fichClase==null)
				return null;
			InputStream input = new FileInputStream(fichClase);
			ByteArrayOutputStream buffer = new ByteArrayOutputStream();
			int data = input.read();

			while(data != -1){
				buffer.write(data);
				data = input.read();
			}

			input.close();

			byte[] classData = buffer.toByteArray();

			return defineClass(nombreClase,
					classData, 0, classData.length);

		} catch (MalformedURLException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace(); 
		}

		return null;
	}

	/*
	     public static void main(String[] args) throws
    ClassNotFoundException,
    IllegalAccessException,
    InstantiationException {

    ClassLoader parentClassLoader = MyClassLoader.class.getClassLoader();
    MyClassLoader classLoader = new MyClassLoader(parentClassLoader);
    Class myObjectClass = classLoader.loadClass("reflection.MyObject");

    AnInterface2       object1 =
            (AnInterface2) myObjectClass.newInstance();

    MyObjectSuperClass object2 =
            (MyObjectSuperClass) myObjectClass.newInstance();

    //create new class loader so classes can be reloaded.
    classLoader = new MyClassLoader(parentClassLoader);
    myObjectClass = classLoader.loadClass("reflection.MyObject");

    object1 = (AnInterface2)       myObjectClass.newInstance();
    object2 = (MyObjectSuperClass) myObjectClass.newInstance();

}

	 */

}
